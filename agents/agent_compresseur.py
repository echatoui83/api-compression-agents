import os
import json
from datetime import datetime
from PIL import Image, ImageOps

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_DISPONIBLE = True
except ImportError:
    HEIF_DISPONIBLE = False

try:
    import pillow_avif
    AVIF_DISPONIBLE = True
except ImportError:
    AVIF_DISPONIBLE = False


class AgentCompresseur:
    """
    Agent qui compresse les images dans differents formats
    selon les recommandations de l Agent 2 (Classifier).
    Supporte : JPEG, PNG, WEBP, HEIF, AVIF
    Version 2.2 : redimensionnement cible 1920x1080
    """

    # ── Dimension cible pour toutes les images ───────────────
    # Mettre None pour garder la taille originale
    DIMENSION_CIBLE = None

    def __init__(self):
        self.nom     = "AgentCompresseur"
        self.version = "2.2"

        self.formats_supportes = ["JPEG", "PNG", "WEBP"]
        if HEIF_DISPONIBLE:
            self.formats_supportes.append("HEIF")
        if AVIF_DISPONIBLE:
            self.formats_supportes.append("AVIF")

        print(f"Agent {self.nom} v{self.version} initialise !")
        print(f"Formats supportes : {self.formats_supportes}")
        if self.DIMENSION_CIBLE:
            print(f"Dimension cible   : {self.DIMENSION_CIBLE[0]}x{self.DIMENSION_CIBLE[1]}")

    # ------------------------------------------------------------------
    def compresser(self, chemin_image, recommandation, dossier_sortie):
        print(f"Compression de : {os.path.basename(chemin_image)}")

        if not os.path.exists(chemin_image):
            return {"statut": "erreur", "message": f"Image non trouvee : {chemin_image}"}

        os.makedirs(dossier_sortie, exist_ok=True)

        image_originale = Image.open(chemin_image)

        # Conversion RGB si necessaire
        if image_originale.mode in ["RGBA", "P", "LA"]:
            fond_blanc = Image.new("RGB", image_originale.size, (255, 255, 255))
            if image_originale.mode == "RGBA":
                fond_blanc.paste(image_originale, mask=image_originale.split()[3])
            else:
                fond_blanc.paste(image_originale)
            image_originale = fond_blanc
        elif image_originale.mode != "RGB":
            image_originale = image_originale.convert("RGB")

        largeur_orig, hauteur_orig = image_originale.size

        # ── Redimensionnement 4K si trop grande ──────────────
        MAX_PIXELS = 4000 * 4000
        nb_pixels  = largeur_orig * hauteur_orig

        if nb_pixels > MAX_PIXELS:
            ratio            = (MAX_PIXELS / nb_pixels) ** 0.5
            nouvelle_largeur = int(largeur_orig * ratio)
            nouvelle_hauteur = int(hauteur_orig * ratio)
            image_originale  = image_originale.resize(
                (nouvelle_largeur, nouvelle_hauteur), Image.LANCZOS
            )
            print(f"  Redim 4K : {largeur_orig}x{hauteur_orig} "
                  f"→ {nouvelle_largeur}x{nouvelle_hauteur}")
            largeur_orig, hauteur_orig = nouvelle_largeur, nouvelle_hauteur

        # ── Redimensionnement à la dimension cible ────────────
        # ImageOps.fit redimensionne ET recadre au centre
        # → pas de déformation, ratio conservé
        if self.DIMENSION_CIBLE is not None:
            target_w, target_h = self.DIMENSION_CIBLE
            if image_originale.size != (target_w, target_h):
                image_originale = ImageOps.fit(
                    image_originale,
                    (target_w, target_h),
                    method    = Image.LANCZOS,
                    centering = (0.5, 0.5)
                )
                print(f"  Redim cible : {largeur_orig}x{hauteur_orig} "
                      f"→ {target_w}x{target_h} (recadrage centre)")

        taille_originale_kb = os.path.getsize(chemin_image) / 1024
        nom_base            = os.path.splitext(os.path.basename(chemin_image))[0]

        format_rec  = str(recommandation.get("format_recommande", "JPEG")).upper()
        qualite_rec = recommandation.get("qualite_recommandee", 85)
        progressive = recommandation.get("parametres_avances", {}).get("progressive", True)

        resultats_compression = []

        # COMPRESSION 1 : Format recommande par le LLM — PRIORITE ABSOLUE
        print(f"  Compression prioritaire LLM : {format_rec} q={qualite_rec}%")
        resultats_compression.append(self._compresser_format(
            image               = image_originale,
            nom_base            = nom_base,
            format_cible        = format_rec,
            qualite             = qualite_rec,
            progressive         = progressive,
            dossier_sortie      = dossier_sortie,
            taille_originale_kb = taille_originale_kb,
            label               = "recommande"
        ))

        fast_mode = os.environ.get("FAST_COMPRESSION") == "1"

        # COMPRESSION 2 : JPEG qualite 85 (comparaison)
        if not fast_mode and format_rec != "JPEG":
            resultats_compression.append(self._compresser_format(
                image               = image_originale,
                nom_base            = nom_base,
                format_cible        = "JPEG",
                qualite             = 85,
                progressive         = True,
                dossier_sortie      = dossier_sortie,
                taille_originale_kb = taille_originale_kb,
                label               = "jpeg_85"
            ))

        # COMPRESSION 3 : WEBP qualite 80 (comparaison)
        if not fast_mode and format_rec != "WEBP":
            resultats_compression.append(self._compresser_format(
                image               = image_originale,
                nom_base            = nom_base,
                format_cible        = "WEBP",
                qualite             = 80,
                progressive         = False,
                dossier_sortie      = dossier_sortie,
                taille_originale_kb = taille_originale_kb,
                label               = "webp_80"
            ))

        # COMPRESSION 4 : PNG sans perte (comparaison)
        if not fast_mode and format_rec != "PNG":
            resultats_compression.append(self._compresser_format(
                image               = image_originale,
                nom_base            = nom_base,
                format_cible        = "PNG",
                qualite             = 95,
                progressive         = False,
                dossier_sortie      = dossier_sortie,
                taille_originale_kb = taille_originale_kb,
                label               = "png_lossless"
            ))

        # COMPRESSION 5 : HEIF
        if not fast_mode and HEIF_DISPONIBLE:
            label_heif = "recommande" if format_rec == "HEIF" else "heif_80"
            resultats_compression.append(self._compresser_format(
                image               = image_originale,
                nom_base            = nom_base,
                format_cible        = "HEIF",
                qualite             = qualite_rec if format_rec == "HEIF" else 80,
                progressive         = False,
                dossier_sortie      = dossier_sortie,
                taille_originale_kb = taille_originale_kb,
                label               = label_heif
            ))
        else:
            print("  HEIF non disponible - pillow-heif non installe")

        # COMPRESSION 6 : AVIF — seulement si pas déjà le format recommandé
        if not fast_mode and AVIF_DISPONIBLE and format_rec != "AVIF":
            resultats_compression.append(self._compresser_format(
                image               = image_originale,
                nom_base            = nom_base,
                format_cible        = "AVIF",
                qualite             = 80,
                progressive         = False,
                dossier_sortie      = dossier_sortie,
                taille_originale_kb = taille_originale_kb,
                label               = "avif_80"
            ))
        elif not fast_mode and not AVIF_DISPONIBLE:
            print("  AVIF non disponible - pillow-avif non installe")
        # ── Choisir le meilleur résultat ─────────────────────
        succes   = [c for c in resultats_compression if c.get("statut") == "succes"]
        meilleur = None

        for comp in succes:
            if comp.get("label") == "recommande":
                meilleur = comp
                meilleur["choix_source"] = "llm_recommande"
                print(f"  Meilleur choisi : {meilleur['format']} (recommandation LLM)")
                break

        if not meilleur and succes:
            meilleur = max(succes, key=lambda x: x.get("taux_compression_pct", 0))
            meilleur["choix_source"] = "meilleur_taux_fallback"
            print(f"  Meilleur choisi : {meilleur['format']} (fallback taux)")

        if not meilleur:
            meilleur = resultats_compression[0]
            meilleur["choix_source"] = "premier_disponible"

        rapport = {
            "agent"                : self.nom,
            "date_compression"     : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_originale"      : chemin_image,
            "taille_originale_kb"  : round(taille_originale_kb, 2),
            "dimension_originale"  : f"{largeur_orig}x{hauteur_orig}",
            "dimension_sortie"     : f"{self.DIMENSION_CIBLE[0]}x{self.DIMENSION_CIBLE[1]}" if self.DIMENSION_CIBLE else f"{largeur_orig}x{hauteur_orig}",
            "format_recommande"    : format_rec,
            "qualite_recommandee"  : qualite_rec,
            "compressions"         : [meilleur] if meilleur else resultats_compression,
            "_toutes_compressions" : resultats_compression,
            "meilleure_compression": meilleur,
            "statut"               : "succes"
        }

        print(f"Compression terminee ! Meilleur : {meilleur.get('format')} "
              f"({meilleur.get('taux_compression_pct')}% de reduction) "
              f"[source: {meilleur.get('choix_source')}]")

        return rapport

    # ------------------------------------------------------------------
    def _compresser_format(self, image, nom_base, format_cible, qualite,
                           progressive, dossier_sortie, taille_originale_kb, label):
        extension = format_cible.lower()
        if extension == "jpeg":
            extension = "jpg"

        nom_sortie    = f"{nom_base}_{label}.{extension}"
        chemin_sortie = os.path.join(dossier_sortie, nom_sortie)

        try:
            if format_cible == "JPEG":
                image.save(chemin_sortie, format="JPEG", quality=qualite,
                           optimize=True, progressive=progressive)
            elif format_cible == "PNG":
                image.save(chemin_sortie, format="PNG", optimize=True, compress_level=6)
            elif format_cible == "WEBP":
                image.save(chemin_sortie, format="WEBP", quality=qualite,
                           optimize=True, lossless=False)
            elif format_cible == "HEIF":
                image.save(chemin_sortie, format="HEIF", quality=qualite)
            elif format_cible == "AVIF":
                image.save(chemin_sortie, format="AVIF", quality=qualite)

            taille_compresse_kb  = os.path.getsize(chemin_sortie) / 1024
            taux_compression_pct = round(
                (1 - taille_compresse_kb / taille_originale_kb) * 100, 2
            )
            ratio_compression    = round(taille_originale_kb / taille_compresse_kb, 2)

            print(f"  [{format_cible} q={qualite}] "
                  f"{round(taille_originale_kb)}KB → {round(taille_compresse_kb)}KB "
                  f"({taux_compression_pct}% reduit)")

            return {
                "label"               : label,
                "format"              : format_cible,
                "qualite"             : qualite,
                "chemin_fichier"      : chemin_sortie,
                "taille_originale_kb" : round(taille_originale_kb, 2),
                "taille_compresse_kb" : round(taille_compresse_kb, 2),
                "taux_compression_pct": taux_compression_pct,
                "ratio_compression"   : ratio_compression,
                "statut"              : "succes"
            }

        except Exception as e:
            print(f"  Erreur compression {format_cible} : {e}")
            return {
                "label"  : label,
                "format" : format_cible,
                "statut" : f"erreur: {str(e)}"
            }

    # ------------------------------------------------------------------
    def sauvegarder_rapport(self, rapport, chemin_sortie):
        os.makedirs(os.path.dirname(chemin_sortie), exist_ok=True)
        with open(chemin_sortie, "w", encoding="utf-8") as f:
            json.dump(rapport, f, indent=4, ensure_ascii=False)
        print(f"Rapport sauvegarde : {chemin_sortie}")