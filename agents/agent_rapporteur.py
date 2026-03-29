import os
import json
from datetime import datetime

# ============================================================
# AGENT 5 : RAPPORTEUR
# Role : Rassembler tous les resultats des agents precedents
#        et generer un rapport final complet et lisible
# Entree : Rapports de tous les agents (1, 2, 3, 4)
# Sortie : Rapport final JSON + Resume textuel
# ============================================================

class AgentRapporteur:
    """
    Agent qui collecte tous les resultats du pipeline
    et produit un rapport final structure et lisible.
    """

    def __init__(self):
        self.nom     = "AgentRapporteur"
        self.version = "1.0"
        print(f"Agent {self.nom} v{self.version} initialise !")


    # ============================================================
    # METHODE PRINCIPALE : generer_rapport()
    # ============================================================

    def generer_rapport(self,
                        rapport_analyse,
                        recommandation,
                        rapport_compression,
                        rapport_evaluation):
        """
        Genere le rapport final en combinant tous les rapports.

        Parametres:
            rapport_analyse     : rapport de l Agent 1
            recommandation      : rapport de l Agent 2
            rapport_compression : rapport de l Agent 3
            rapport_evaluation  : rapport de l Agent 4

        Retourne:
            dict : rapport final complet
        """

        print(f"Generation du rapport final...")

        # Recuperer les infos importantes de chaque agent
        meta        = rapport_analyse.get("metadonnees", {})
        complexite  = rapport_analyse.get("complexite", {})
        couleurs    = rapport_analyse.get("couleurs", {})
        meilleure   = rapport_evaluation.get("meilleure_compression", {})
        evaluations = rapport_evaluation.get("evaluations", [])

        # ── Calculer les statistiques globales ──────────────────

        # Taux de compression moyen de toutes les compressions
        taux_liste = [e.get("taux_compression_pct", 0) for e in evaluations]
        taux_moyen = round(sum(taux_liste) / len(taux_liste), 2) if taux_liste else 0

        # PSNR moyen
        psnr_liste = [e.get("psnr_db", 0) for e in evaluations]
        psnr_moyen = round(sum(psnr_liste) / len(psnr_liste), 2) if psnr_liste else 0

        # SSIM moyen
        ssim_liste = [e.get("ssim", 0) for e in evaluations]
        ssim_moyen = round(sum(ssim_liste) / len(ssim_liste), 4) if ssim_liste else 0

        # ── Determiner si la mission est reussie ────────────────
        # La compression est reussie si :
        # - PSNR >= 30 dB (qualite acceptable)
        # - SSIM >= 0.85 (similarite acceptable)
        # - Taux >= 20% (reduction significative)

        psnr_ok = meilleure.get("psnr_db", 0) >= 28   # adapté images palette
        ssim_ok = meilleure.get("ssim", 0) >= 0.80 
        taux_ok = meilleure.get("taux_compression_pct", 0) >= 20

        mission_reussie = psnr_ok and ssim_ok and taux_ok

        # ── Generer le resume textuel ───────────────────────────
        resume = self._generer_resume(
            meta            = meta,
            complexite      = complexite,
            recommandation  = recommandation,
            meilleure       = meilleure,
            taux_moyen      = taux_moyen,
            mission_reussie = mission_reussie
        )

        # ── Assembler le rapport final ──────────────────────────
        rapport_final = {
            "rapport_final"  : True,
            "agent"          : self.nom,
            "date_rapport"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "statut_global"  : "succes" if mission_reussie else "attention",

            # Infos sur l image
            "image" : {
                "nom"          : meta.get("nom_fichier", ""),
                "categorie"    : rapport_analyse.get("categorie", ""),
                "format_original": meta.get("format", ""),
                "resolution"   : meta.get("resolution", ""),
                "taille_originale_kb": meta.get("taille_kb", 0),
                "complexite"   : complexite.get("niveau_complexite", ""),
                "score_complexite": complexite.get("score_complexite", 0),
            },

            # Recommandation du LLM
            "recommandation_llm" : {
                "type_detecte"        : recommandation.get("type_image", ""),
                "format_recommande"   : recommandation.get("format_recommande", ""),
                "qualite_recommandee" : recommandation.get("qualite_recommandee", 0),
                "justification"       : recommandation.get("justification", ""),
                "priorite"            : recommandation.get("priorite", ""),
                "statut_llm"          : recommandation.get("statut", ""),
            },

            # Resultats de compression
            "resultats_compression" : {
                "nb_formats_testes"  : len(evaluations),
                "taux_moyen_pct"     : taux_moyen,
                "psnr_moyen_db"      : psnr_moyen,
                "ssim_moyen"         : ssim_moyen,
                "meilleur_format"    : meilleure.get("format", ""),
                "meilleure_qualite"  : meilleure.get("qualite", 0),
                "meilleur_taux_pct"  : meilleure.get("taux_compression_pct", 0),
                "meilleur_psnr_db"   : meilleure.get("psnr_db", 0),
                "meilleur_ssim"      : meilleure.get("ssim", 0),
                "meilleur_score"     : meilleure.get("score_global", 0),
                "taille_finale_kb"   : meilleure.get("taille_compresse_kb", 0),
                "economie_kb"        : round(
                    meta.get("taille_kb", 0) - meilleure.get("taille_compresse_kb", 0), 2
                ),
            },

            # Criteres de validation
            "validation" : {
                "psnr_acceptable"  : psnr_ok,
                "ssim_acceptable"  : ssim_ok,
                "compression_suffisante": taux_ok,
                "mission_reussie"  : mission_reussie,
            },

            # Tableau de toutes les evaluations
            "detail_evaluations" : evaluations,

            # Resume lisible
            "resume" : resume,
        }

        print(f"Rapport final genere !")
        print(f"Statut : {'SUCCES' if mission_reussie else 'ATTENTION'}")
        print(f"Meilleur format : {meilleure.get('format')} "
              f"(PSNR={meilleure.get('psnr_db')} dB, "
              f"SSIM={meilleure.get('ssim')}, "
              f"Taux={meilleure.get('taux_compression_pct')}%)")

        return rapport_final


    # ============================================================
    # METHODE : _generer_resume()
    # Genere un resume textuel lisible par un humain
    # ============================================================

    def _generer_resume(self, meta, complexite, recommandation,
                        meilleure, taux_moyen, mission_reussie):
        """
        Genere un resume en texte clair et lisible.
        C est ce qui sera affiche dans le terminal.
        """

        statut_emoji = "SUCCES" if mission_reussie else "ATTENTION"

        resume = f"""
========================================================
RAPPORT FINAL DE COMPRESSION
========================================================
Statut        : {statut_emoji}
Date          : {datetime.now().strftime("%d/%m/%Y %H:%M")}

IMAGE ANALYSEE :
  Fichier     : {meta.get("nom_fichier", "")}
  Resolution  : {meta.get("resolution", "")}
  Taille orig.: {meta.get("taille_kb", 0)} KB
  Complexite  : {complexite.get("niveau_complexite", "").upper()}

RECOMMANDATION IA (Groq/Llama3) :
  Type image  : {recommandation.get("type_image", "")}
  Format rec. : {recommandation.get("format_recommande", "")}
  Qualite rec.: {recommandation.get("qualite_recommandee", "")}%
  Justif.     : {recommandation.get("justification", "")[:100]}...

MEILLEURE COMPRESSION :
  Format      : {meilleure.get("format", "")} qualite={meilleure.get("qualite", "")}%
  Taille fin. : {meilleure.get("taille_compresse_kb", 0):.1f} KB
  Taux comp.  : {meilleure.get("taux_compression_pct", 0)}%
  PSNR        : {meilleure.get("psnr_db", 0)} dB
  SSIM        : {meilleure.get("ssim", 0)}
  Score global: {meilleure.get("score_global", 0)}/100

CONCLUSION :
  La compression {meilleure.get("format", "")} avec une qualite de
  {meilleure.get("qualite", "")}% offre le meilleur equilibre entre
  qualite visuelle et reduction de taille.
  Reduction obtenue : {meilleure.get("taux_compression_pct", 0)}%
========================================================
"""
        return resume


    def sauvegarder_rapport(self, rapport, chemin_sortie):
        """Sauvegarde le rapport final en JSON."""
        os.makedirs(os.path.dirname(chemin_sortie), exist_ok=True)
        with open(chemin_sortie, "w", encoding="utf-8") as f:
            json.dump(rapport, f, indent=4, ensure_ascii=False)
        print(f"Rapport sauvegarde : {chemin_sortie}")


    def afficher_resume(self, rapport):
        """Affiche le resume dans le terminal."""
        print(rapport.get("resume", "Pas de resume disponible"))
