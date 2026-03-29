
import os
from dotenv import load_dotenv
import numpy as np
from PIL import Image
import cv2
from scipy.stats import entropy
from skimage.feature import graycomatrix, graycoprops
import pytesseract

load_dotenv()
tesseract_path = os.getenv("TESSERACT_CMD", "C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
pytesseract.pytesseract.tesseract_cmd = tesseract_path

import json
from datetime import datetime


class AgentAnalyseur:
    """
    Agent responsable de l analyse complete d une image.
    Extrait : metadonnees, couleurs, complexite, textures GLCM, OCR.
    Version : 2.0
    """

    def __init__(self):
        self.nom     = "AgentAnalyseur"
        self.version = "2.0"
        print(f"Agent {self.nom} v{self.version} initialise et pret !")

    def analyser(self, chemin_image):
        print(f"Analyse de : {chemin_image}")

        if not os.path.exists(chemin_image):
            return {
                "statut" : "erreur",
                "message": f"Image non trouvee : {chemin_image}"
            }

        try:
            image_pil = Image.open(chemin_image)
            image_cv2 = cv2.imread(chemin_image)
            if image_cv2 is None:
                raise ValueError("Impossible de lire l'image (fichier corrompu ou format non supporte par OpenCV).")
        except Exception as e:
            return {
                "statut" : "erreur",
                "message": str(e)
            }

        metadonnees = self._extraire_metadonnees(chemin_image, image_pil)
        couleurs    = self._analyser_couleurs(image_pil)
        complexite  = self._analyser_complexite(image_cv2)
        textures    = self._analyser_textures_glcm(image_cv2)
        ocr         = self._detecter_texte_ocr(image_pil)

        rapport = {
            "agent"       : self.nom,
            "date_analyse": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image"       : chemin_image,
            "statut"      : "succes",
            "metadonnees" : metadonnees,
            "couleurs"    : couleurs,
            "complexite"  : complexite,
            "textures"    : textures,
            "ocr"         : ocr,
        }

        print(f"Analyse terminee pour : {chemin_image}")
        return rapport

    def _extraire_metadonnees(self, chemin_image, image_pil):
        taille_kb        = os.path.getsize(chemin_image) / 1024
        largeur, hauteur = image_pil.size
        mode             = image_pil.mode
        canaux           = len(image_pil.getbands())
        _, extension     = os.path.splitext(chemin_image)
        return {
            "nom_fichier" : os.path.basename(chemin_image),
            "format"      : extension.upper().replace(".", ""),
            "largeur_px"  : largeur,
            "hauteur_px"  : hauteur,
            "resolution"  : f"{largeur}x{hauteur}",
            "taille_kb"   : round(taille_kb, 2),
            "taille_mb"   : round(taille_kb / 1024, 3),
            "mode_couleur": mode,
            "nb_canaux"   : canaux,
            "nb_pixels"   : largeur * hauteur,
        }

    def _analyser_couleurs(self, image_pil):
        pixels  = np.array(image_pil.convert("RGB"))
        canal_r = pixels[:, :, 0]
        canal_g = pixels[:, :, 1]
        canal_b = pixels[:, :, 2]
        return {
            "rouge": {
                "moyenne"   : round(float(np.mean(canal_r)), 2),
                "ecart_type": round(float(np.std(canal_r)),  2),
                "min"       : int(np.min(canal_r)),
                "max"       : int(np.max(canal_r)),
            },
            "vert": {
                "moyenne"   : round(float(np.mean(canal_g)), 2),
                "ecart_type": round(float(np.std(canal_g)),  2),
                "min"       : int(np.min(canal_g)),
                "max"       : int(np.max(canal_g)),
            },
            "bleu": {
                "moyenne"   : round(float(np.mean(canal_b)), 2),
                "ecart_type": round(float(np.std(canal_b)),  2),
                "min"       : int(np.min(canal_b)),
                "max"       : int(np.max(canal_b)),
            },
            "luminosite_globale": round(float(np.mean(pixels)), 2),
        }

    def _analyser_complexite(self, image_cv2):
        gris            = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
        histogramme, _  = np.histogram(gris, bins=256, range=(0, 256), density=True)
        histogramme     = histogramme + 1e-10
        entropie_valeur = round(float(entropy(histogramme)), 4)

        contours          = cv2.Canny(gris, threshold1=50, threshold2=150)
        nb_pixels_total   = gris.shape[0] * gris.shape[1]
        nb_pixels_contour = np.sum(contours > 0)
        ratio_contours    = round(float(nb_pixels_contour / nb_pixels_total) * 100, 4)

        gradient_x         = cv2.Sobel(gris, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y         = cv2.Sobel(gris, cv2.CV_64F, 0, 1, ksize=3)
        magnitude_gradient = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_moyen     = round(float(np.mean(magnitude_gradient)), 4)

        score_entropie = min(entropie_valeur / 8.0,   1.0)
        score_contours = min(ratio_contours  / 20.0,  1.0)
        score_gradient = min(gradient_moyen  / 100.0, 1.0)
        score_global   = round(
            score_entropie * 0.4 + score_contours * 0.3 + score_gradient * 0.3, 4
        )

        if score_global < 0.3:
            niveau = "faible"
        elif score_global < 0.6:
            niveau = "Modéré"
        else:
            niveau = "eleve"

        return {
            "entropie"          : entropie_valeur,
            "ratio_contours_pct": ratio_contours,
            "gradient_moyen"    : gradient_moyen,
            "score_complexite"  : score_global,
            "niveau_complexite" : niveau,
        }

    def _analyser_textures_glcm(self, image_cv2):
        """
        Analyse les textures via la matrice de co-occurrence GLCM.
        Extrait : contraste, homogeneite, energie, correlation.
        """
        try:
            gris = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)

            h, w = gris.shape
            if h > 512 or w > 512:
                gris = cv2.resize(gris, (512, 512))

            gris_reduit = (gris // 4).astype(np.uint8)

            glcm = graycomatrix(
                gris_reduit,
                distances = [1],
                angles    = [0, np.pi/4, np.pi/2, 3*np.pi/4],
                levels    = 64,
                symmetric = True,
                normed    = True
            )

            contraste   = round(float(np.mean(graycoprops(glcm, "contrast"))),    4)
            homogeneite = round(float(np.mean(graycoprops(glcm, "homogeneity"))), 4)
            energie     = round(float(np.mean(graycoprops(glcm, "energy"))),      4)
            correlation = round(float(np.mean(graycoprops(glcm, "correlation"))), 4)

            return {
                "contraste"  : contraste,
                "homogeneite": homogeneite,
                "energie"    : energie,
                "correlation": correlation,
                "statut"     : "succes"
            }

        except Exception as e:
            return {
                "contraste"  : None,
                "homogeneite": None,
                "energie"    : None,
                "correlation": None,
                "statut"     : f"erreur : {str(e)}"
            }

    def _detecter_texte_ocr(self, image_pil):
        """
        Detecte la presence de texte dans l image via OCR (pytesseract).
        Retourne le texte trouve et un score de confiance.
        """
        try:
            largeur, hauteur = image_pil.size
            if largeur > 1500 or hauteur > 1500:
                image_pil = image_pil.resize((1500, 1500), Image.LANCZOS)

            image_gris = image_pil.convert("L")

            texte_brut    = pytesseract.image_to_string(image_gris, lang="fra+eng")
            texte_nettoye = texte_brut.strip()
            mots          = [m for m in texte_nettoye.split() if len(m) > 1]
            nb_mots       = len(mots)

            donnees = pytesseract.image_to_data(
                image_gris,
                lang        = "fra+eng",
                output_type = pytesseract.Output.DICT
            )
            confidences = [
                int(c) for c in donnees["conf"]
                if str(c).isdigit() and int(c) > 0
            ]
            confiance_moyenne = round(
                sum(confidences) / len(confidences), 1
            ) if confidences else 0.0

            texte_detecte = nb_mots >= 3 and confiance_moyenne >= 40

            if texte_detecte and nb_mots > 20:
                suggestion = "document"
            elif texte_detecte:
                suggestion = "screenshot"
            else:
                suggestion = "photo"

            return {
                "texte_detecte"    : texte_detecte,
                "nb_mots"          : nb_mots,
                "confiance_moyenne": confiance_moyenne,
                "extrait_texte"    : texte_nettoye[:200],
                "suggestion_type"  : suggestion,
                "statut"           : "succes"
            }

        except Exception as e:
            return {
                "texte_detecte"    : False,
                "nb_mots"          : 0,
                "confiance_moyenne": 0.0,
                "extrait_texte"    : "",
                "suggestion_type"  : "inconnu",
                "statut"           : f"erreur : {str(e)}"
            }

    def sauvegarder_rapport(self, rapport, chemin_sortie):
        os.makedirs(os.path.dirname(chemin_sortie), exist_ok=True)
        with open(chemin_sortie, "w", encoding="utf-8") as f:
            json.dump(rapport, f, indent=4, ensure_ascii=False)
        print(f"Rapport sauvegarde : {chemin_sortie}")