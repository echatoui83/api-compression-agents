import os
import json
import numpy as np
from datetime import datetime
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


class AgentEvaluateur:
    """
    Agent qui calcule les metriques de qualite pour evaluer
    l impact de la compression sur une image.

    Metriques implementees :
    - PSNR  : Peak Signal-to-Noise Ratio (en dB)
              >= 40 dB = excellent
              35-40 dB = tres_bon
              28-35 dB = acceptable
              24-28 dB = degradation_visible
              < 24 dB  = mauvais

    - SSIM  : Structural Similarity Index (entre 0 et 1)
              >= 0.95 = excellent
              0.90-0.95 = tres_bon
              0.80-0.90 = acceptable
              < 0.80 = mauvais

    - MSE   : Mean Squared Error
              0 = images identiques

    - Taux  : Pourcentage de reduction de taille
              formule : (1 - taille_compresse/taille_originale) x 100

    Version 1.1 :
    - Respecte la recommandation LLM en priorite
    - Retourne meilleure_compression (LLM) + meilleure_par_metriques
    - Coherence LLM vs metriques calculee
    - Seuils PSNR adaptes aux images palette (>= 28 au lieu de 30)
    """

    def __init__(self):
        self.nom     = "AgentEvaluateur"
        self.version = "1.1"
        print(f"Agent {self.nom} v{self.version} initialise !")

    # ------------------------------------------------------------------
    def evaluer(self, chemin_originale, rapport_compression):
        """
        Evalue la qualite de toutes les compressions effectuees.

        Parametres:
            chemin_originale    : chemin vers l image originale
            rapport_compression : rapport JSON de l Agent 3

        Retourne:
            dict : rapport complet avec toutes les metriques
        """
        print(f"Evaluation de : {os.path.basename(chemin_originale)}")

        # Charger l image originale
        image_orig          = np.array(Image.open(chemin_originale).convert("RGB"))
        taille_originale_kb = os.path.getsize(chemin_originale) / 1024
        evaluations         = []
        # On utilise le tableau cache s'il existe (contient toutes les compressions), sinon l'officiel
        compressions        = rapport_compression.get("_toutes_compressions", rapport_compression.get("compressions", []))

        for compression in compressions:
            if compression.get("statut") != "succes":
                continue

            chemin_compresse = compression.get("chemin_fichier", "")
            if not os.path.exists(chemin_compresse):
                print(f"  Fichier non trouve : {chemin_compresse}")
                continue

            metriques = self._calculer_metriques(
                image_orig          = image_orig,
                chemin_compresse    = chemin_compresse,
                taille_originale_kb = taille_originale_kb,
                compression         = compression
            )
            evaluations.append(metriques)
            print(f"  [{compression['format']} q={compression['qualite']}] "
                  f"PSNR={metriques['psnr_db']} dB | "
                  f"SSIM={metriques['ssim']} | "
                  f"Taux={metriques['taux_compression_pct']}%")

        # ── Meilleure compression selon LLM ──────────────────────────
        format_rec          = rapport_compression.get("format_recommande")
        meilleure           = self._trouver_meilleure(evaluations, format_rec)

        # ── Meilleure compression selon metriques ────────────────────
        meilleure_metriques = self._trouver_meilleure_metriques(evaluations)

        # ── Coherence LLM vs Metriques ───────────────────────────────
        coherence = (
            meilleure.get("format") == meilleure_metriques.get("format")
            if meilleure and meilleure_metriques else False
        )

        rapport = {
            "agent"                   : self.nom,
            "version"                 : self.version,
            "date_evaluation"         : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_originale"         : chemin_originale,
            "taille_originale_kb"     : round(taille_originale_kb, 2),
            "evaluations"             : [e for e in evaluations if e.get("label") == "recommande"] if evaluations else evaluations,
            "_toutes_evaluations"     : evaluations,
            "meilleure_compression"   : meilleure,
            "meilleure_par_metriques" : meilleure_metriques,
            "coherence_llm_metriques" : coherence,
            "statut"                  : "succes"
        }

        # ── Nettoyage automatique des fichiers de comparaison ────────
        # On supprime physiquement les fichiers de comparaison sur le système
        # après génération des graphes pour s'assurer que n8n ne puisse pas les récupérer
        chemin_meilleure = meilleure.get("chemin_fichier") if meilleure else None
        
        for compression in compressions:
            chem = compression.get("chemin_fichier")
            if chem and chem != chemin_meilleure:
                if os.path.exists(chem):
                    try:
                        os.remove(chem)
                    except Exception:
                        pass
                        
        print(f"Evaluation terminee !")
        if meilleure:
            print(f"  Meilleur LLM       : {meilleure.get('format')} "
                  f"(source: {meilleure.get('choix_source', 'N/A')})")
        if meilleure_metriques:
            print(f"  Meilleur metriques : {meilleure_metriques.get('format')} "
                  f"(score={meilleure_metriques.get('score_global')})")
        print(f"  Coherence LLM/Metriques : {coherence}")

        return rapport

    # ------------------------------------------------------------------
    def _trouver_meilleure(self, evaluations, format_recommande=None):
        """
        Trouve la meilleure compression.
        Priorite 1 : format recommande par le LLM (label='recommande')
        Priorite 2 : meilleur score global mathematique
        """
        if not evaluations:
            return None

        # Exclure les compressions avec taux nul (PNG sans perte souvent 0)
        succes = [e for e in evaluations if e.get("taux_compression_pct", 0) != 0]
        if not succes:
            return evaluations[0] if evaluations else None

        # Priorite au format LLM avec label "recommande"
        if format_recommande:
            for e in succes:
                if (e.get("format") == format_recommande and
                        e.get("label") == "recommande"):
                    e["choix_source"] = "llm_recommande"
                    return e

        # Fallback : meilleur score global
        meilleur = max(succes, key=lambda x: x.get("score_global", 0))
        meilleur["choix_source"] = "metriques_fallback"
        return meilleur

    # ------------------------------------------------------------------
    def _trouver_meilleure_metriques(self, evaluations):
        """
        Trouve la meilleure compression selon le score mathematique pur.
        Independante de la recommandation LLM.
        """
        if not evaluations:
            return None

        succes = [e for e in evaluations if e.get("taux_compression_pct", 0) != 0]
        if not succes:
            return evaluations[0]

        meilleur = max(succes, key=lambda x: x.get("score_global", 0))
        meilleur["choix_source"] = "metriques"
        return meilleur

    # ------------------------------------------------------------------
    def _calculer_metriques(self, image_orig, chemin_compresse,
                             taille_originale_kb, compression):
        """
        Calcule toutes les metriques de qualite pour une image compressee.
        """
        # Charger l image compressee
        image_comp = np.array(Image.open(chemin_compresse).convert("RGB"))

        # Redimensionner si les tailles sont differentes
        if image_orig.shape != image_comp.shape:
            from PIL import Image as PILImage
            img_pil    = PILImage.fromarray(image_comp)
            img_pil    = img_pil.resize(
                (image_orig.shape[1], image_orig.shape[0]),
                PILImage.LANCZOS
            )
            image_comp = np.array(img_pil)

        # ── MSE : Mean Squared Error ──────────────────────────────────
        # Formule : MSE = moyenne( (pixel_orig - pixel_comp)^2 )
        # MSE = 0 → images identiques
        mse_valeur = round(float(np.mean(
            (image_orig.astype(float) - image_comp.astype(float)) ** 2
        )), 4)

        # ── PSNR : Peak Signal-to-Noise Ratio ────────────────────────
        # Formule : PSNR = 10 x log10(255^2 / MSE)
        # Plus c est eleve, meilleure est la qualite
        if mse_valeur == 0:
            psnr_valeur = 100.0  # images identiques → PSNR infini, on met 100
        else:
            psnr_valeur = round(float(psnr(image_orig, image_comp, data_range=255)), 4)

        # ── SSIM : Structural Similarity Index ───────────────────────
        # Mesure la similarite structurelle (luminosite, contraste, structure)
        # Valeur entre 0 et 1 : 1.0 = identiques
        ssim_valeur = round(float(ssim(
            image_orig, image_comp,
            channel_axis = 2,    # axe des canaux RGB
            data_range   = 255   # plage de valeurs 0-255
        )), 4)

        # ── Taux de compression ───────────────────────────────────────
        # Formule : taux = (1 - taille_compresse / taille_originale) x 100
        # Positif = reduction | Negatif = augmentation (ex: PNG sur photo JPEG)
        taille_compresse_kb  = os.path.getsize(chemin_compresse) / 1024
        taux_compression_pct = round(
            (1 - taille_compresse_kb / taille_originale_kb) * 100, 2
        )
        ratio_compression    = round(taille_originale_kb / taille_compresse_kb, 2)

        # ── Score Global : PSNR 40% + SSIM 40% + Taux 20% ────────────
        # Poids : qualite visuelle prime (80%) sur reduction de taille (20%)
        score_psnr   = min(psnr_valeur / 50.0, 1.0)        # normalise /50 dB
        score_ssim   = ssim_valeur                          # deja entre 0 et 1
        score_taux   = min(taux_compression_pct / 80.0, 1.0) # normalise /80%

        score_global = round(
            (score_psnr * 0.4 + score_ssim * 0.4 + score_taux * 0.2) * 100, 2
        )

        # ── Interpretations qualitatives ─────────────────────────────
        # Seuils PSNR adaptes aux images palette (mode P) : >= 28 au lieu de 30
        qualite_psnr = (
            "excellent"          if psnr_valeur >= 40 else
            "tres_bon"           if psnr_valeur >= 35 else
            "acceptable"         if psnr_valeur >= 28 else
            "degradation_visible" if psnr_valeur >= 24 else
            "mauvais"
        )
        qualite_ssim = (
            "excellent"  if ssim_valeur >= 0.95 else
            "tres_bon"   if ssim_valeur >= 0.90 else
            "acceptable" if ssim_valeur >= 0.80 else
            "mauvais"
        )

        return {
            "format"              : compression.get("format"),
            "qualite"             : compression.get("qualite"),
            "label"               : compression.get("label"),
            "chemin_fichier"      : chemin_compresse,
            "taille_originale_kb" : round(taille_originale_kb, 2),
            "taille_compresse_kb" : round(taille_compresse_kb, 2),
            "taux_compression_pct": taux_compression_pct,
            "ratio_compression"   : ratio_compression,
            "mse"                 : mse_valeur,
            "psnr_db"             : psnr_valeur,
            "qualite_psnr"        : qualite_psnr,
            "ssim"                : ssim_valeur,
            "qualite_ssim"        : qualite_ssim,
            "score_global"        : score_global,
        }

    # ------------------------------------------------------------------
    def sauvegarder_rapport(self, rapport, chemin_sortie):
        """Sauvegarde le rapport d evaluation en JSON."""
        os.makedirs(os.path.dirname(chemin_sortie), exist_ok=True)
        with open(chemin_sortie, "w", encoding="utf-8") as f:
            json.dump(rapport, f, indent=4, ensure_ascii=False)
        print(f"Rapport sauvegarde : {chemin_sortie}")