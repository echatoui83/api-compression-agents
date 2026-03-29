import os
import json
import time
from datetime import datetime
from groq import Groq
import cohere
from mistralai.client import Mistral
from dotenv import load_dotenv

class AgentClassifier:

    def __init__(self):
        self.nom     = "AgentClassifier-MultiLLM"
        self.version = "3.2"

        load_dotenv()

        groq_key = os.getenv("GROQ_API_KEY")
        self.groq_client    = Groq(api_key=groq_key)
        self.groq_modele    = "llama-3.3-70b-versatile"

        cohere_key = os.getenv("COHERE_API_KEY")
        self.cohere_client  = cohere.ClientV2(api_key=cohere_key)
        self.cohere_modele  = "command-r-08-2024"

        mistral_key = os.getenv("MISTRAL_API_KEY")
        self.mistral_client = Mistral(api_key=mistral_key)
        self.mistral_modele = "mistral-small-latest"

        print(f"Agent {self.nom} v{self.version} initialise !")
        print(f"  LLM1 : Groq/{self.groq_modele}")
        print(f"  LLM2 : Cohere/{self.cohere_modele}")
        print(f"  LLM3 : Mistral/{self.mistral_modele}")

    # ------------------------------------------------------------------
    def classifier(self, rapport_agent1):
        print(f"Classification Multi-LLM (3 LLMs) : {rapport_agent1.get('image', '')}")

        meta       = rapport_agent1.get("metadonnees", {})
        couleurs   = rapport_agent1.get("couleurs", {})
        complexite = rapport_agent1.get("complexite", {})
        textures   = rapport_agent1.get("textures", {})
        ocr        = rapport_agent1.get("ocr", {})
        categorie  = rapport_agent1.get("categorie", "inconnue")

        usage_map = {
            "photos"     : "web et partage sur reseaux sociaux",
            "documents"  : "archivage et impression professionnelle",
            "graphiques" : "web et presentation technique",
            "screenshots": "documentation et interface web"
        }
        priorite_map = {
            "photos"     : "equilibre qualite/taille avec compatibilite maximale",
            "documents"  : "qualite maximale sans perte pour lisibilite du texte",
            "graphiques" : "fidelite des couleurs et nettete des formes",
            "screenshots": "nettete du texte et des elements d interface"
        }
        format_conseille_map = {
            "photos"     : "JPEG, WEBP ou AVIF",
            "documents"  : "PNG ou WEBP sans perte",
            "graphiques" : "PNG, WEBP ou AVIF",
            "screenshots": "PNG ou WEBP"
        }

        usage            = usage_map.get(categorie, "usage general")
        priorite         = priorite_map.get(categorie, "equilibre qualite/taille")
        format_conseille = format_conseille_map.get(categorie, "JPEG")

        prompt = f"""
Tu es un expert en compression d images.
Analyse ces caracteristiques et recommande la meilleure compression.

INFORMATIONS :
- Categorie          : {categorie}
- Usage prevu        : {usage}
- Priorite           : {priorite}
- Format conseille   : {format_conseille}
- Format actuel      : {meta.get("format", "inconnu")}
- Resolution         : {meta.get("resolution", "inconnue")}
- Taille actuelle    : {meta.get("taille_kb", 0)} KB
- Mode couleur       : {meta.get("mode_couleur", "inconnu")}

ANALYSE VISUELLE :
- Luminosite globale : {couleurs.get("luminosite_globale", 0)} / 255
- Complexite visuelle: {complexite.get("niveau_complexite", "inconnu")}
- Score complexite   : {complexite.get("score_complexite", 0)} / 1.0
- Entropie           : {complexite.get("entropie", 0)}
- Ratio contours     : {complexite.get("ratio_contours_pct", 0)} %

TEXTURES GLCM :
- Contraste          : {textures.get("contraste", "N/A")}
- Homogeneite        : {textures.get("homogeneite", "N/A")}
- Energie            : {textures.get("energie", "N/A")}
- Correlation        : {textures.get("correlation", "N/A")}

DETECTION OCR :
- Texte detecte      : {ocr.get("texte_detecte", False)}
- Nombre de mots     : {ocr.get("nb_mots", 0)}
- Confiance OCR      : {ocr.get("confiance_moyenne", 0)} %
- Suggestion type    : {ocr.get("suggestion_type", "inconnu")}

FORMATS DISPONIBLES POUR LA COMPRESSION :
- JPEG  : compatibilite maximale, ideal photos classiques
- PNG   : sans perte, obligatoire si texte detecte ou document
- WEBP  : bon compromis qualite/taille, web moderne
- HEIF  : haute qualite, ideal photos haute resolution et mobile
- AVIF  : meilleure compression (-50% vs JPEG), ideal photos web modernes

REPONDS UNIQUEMENT EN JSON valide :
{{
    "type_image"              : "photo",
    "format_recommande"       : "JPEG",
    "qualite_recommandee"     : 85,
    "taux_compression_estime" : 70,
    "priorite"                : "equilibre",
    "justification"           : "Explication courte"
}}
"""

        print("  LLM1 : Appel Groq/Llama3...")
        rec1 = self._appeler_groq(prompt, rapport_agent1)
        time.sleep(1)

        print("  LLM2 : Appel Cohere...")
        rec2 = self._appeler_cohere(prompt, rapport_agent1)
        time.sleep(1)

        print("  LLM3 : Appel Mistral...")
        rec3 = self._appeler_mistral(prompt, rapport_agent1)

        meilleure = self._vote_majoritaire(rec1, rec2, rec3, rapport_agent1)

        meilleure["multi_llm"] = {
            "llm1_groq"   : {
                "modele" : self.groq_modele,
                "format" : rec1.get("format_recommande"),
                "qualite": rec1.get("qualite_recommandee"),
                "statut" : rec1.get("statut")
            },
            "llm2_cohere" : {
                "modele" : self.cohere_modele,
                "format" : rec2.get("format_recommande"),
                "qualite": rec2.get("qualite_recommandee"),
                "statut" : rec2.get("statut")
            },
            "llm3_mistral": {
                "modele" : self.mistral_modele,
                "format" : rec3.get("format_recommande"),
                "qualite": rec3.get("qualite_recommandee"),
                "statut" : rec3.get("statut")
            },
            "llm_choisi"  : meilleure.get("llm_source", "vote"),
            "accord"      : self._verifier_accord(rec1, rec2, rec3)
        }

        meilleure["contexte"] = {
            "categorie"       : categorie,
            "usage"           : usage,
            "priorite"        : priorite,
            "format_conseille": format_conseille
        }

        # ← NOUVEAU : stocker clairement le format choisi par les LLMs
        meilleure["format_recommande_llm"] = meilleure.get("format_recommande")
        meilleure["nb_votes_llm"]          = int(
            meilleure.get("llm_source", "1").split("_")[-3]
        ) if "vote_majoritaire" in meilleure.get("llm_source", "") else 3

        print(f"  Groq    : {rec1.get('format_recommande')} q={rec1.get('qualite_recommandee')}%")
        print(f"  Cohere  : {rec2.get('format_recommande')} q={rec2.get('qualite_recommandee')}%")
        print(f"  Mistral : {rec3.get('format_recommande')} q={rec3.get('qualite_recommandee')}%")
        print(f"  Accord  : {meilleure['multi_llm']['accord']}")
        print(f"  Choix   : {meilleure['multi_llm']['llm_choisi']}")
        print(f"  Usage   : {usage}")

        return meilleure

    # ------------------------------------------------------------------
    def _appeler_groq(self, prompt, rapport_agent1):
        try:
            reponse = self.groq_client.chat.completions.create(
                model    = self.groq_modele,
                messages = [
                    {"role": "system", "content": "Reponds UNIQUEMENT en JSON valide."},
                    {"role": "user",   "content": prompt}
                ],
                temperature = 0.1
            )
            texte = reponse.choices[0].message.content.strip()
            rec   = self._extraire_json(texte)
            rec["llm_source"] = "groq"
            rec["statut"]     = "succes"
            return rec
        except Exception as e:
            print(f"  Erreur Groq : {e}")
            return self._recommandation_par_defaut(rapport_agent1, "groq", str(e))

    # ------------------------------------------------------------------
    def _appeler_cohere(self, prompt, rapport_agent1):
        try:
            reponse = self.cohere_client.chat(
                model    = self.cohere_modele,
                messages = [
                    {"role": "system", "content": "Reponds UNIQUEMENT en JSON valide."},
                    {"role": "user",   "content": prompt}
                ]
            )
            texte = reponse.message.content[0].text.strip()
            rec   = self._extraire_json(texte)
            rec["llm_source"] = "cohere"
            rec["statut"]     = "succes"
            return rec
        except Exception as e:
            print(f"  Erreur Cohere : {e}")
            return self._recommandation_par_defaut(rapport_agent1, "cohere", str(e))

    # ------------------------------------------------------------------
    def _appeler_mistral(self, prompt, rapport_agent1):
        try:
            reponse = self.mistral_client.chat.complete(
                model    = self.mistral_modele,
                messages = [
                    {"role": "system", "content": "Reponds UNIQUEMENT en JSON valide."},
                    {"role": "user",   "content": prompt}
                ]
            )
            texte = reponse.choices[0].message.content.strip()
            rec   = self._extraire_json(texte)
            rec["llm_source"] = "mistral"
            rec["statut"]     = "succes"
            return rec
        except Exception as e:
            print(f"  Erreur Mistral : {e}")
            return self._recommandation_par_defaut(rapport_agent1, "mistral", str(e))

    # ------------------------------------------------------------------
    def _extraire_json(self, texte):
        texte = texte.strip()
        debut = texte.find('{')
        fin = texte.rfind('}')
        if debut != -1 and fin != -1 and fin > debut:
            try:
                return json.loads(texte[debut:fin+1])
            except json.JSONDecodeError:
                pass
        
        texte_clean = texte.replace("```json", "").replace("```", "").strip()
        return json.loads(texte_clean)

    # ------------------------------------------------------------------
    def _vote_majoritaire(self, rec1, rec2, rec3, rapport_agent1):
        formats = [
            rec1.get("format_recommande", "JPEG"),
            rec2.get("format_recommande", "JPEG"),
            rec3.get("format_recommande", "JPEG")
        ]
        votes = {}
        for f in formats:
            votes[f] = votes.get(f, 0) + 1

        format_gagnant = max(votes, key=votes.get)
        nb_votes       = votes[format_gagnant]
        print(f"  Votes : {votes} → Gagnant : {format_gagnant} ({nb_votes}/3)")

        for rec in [rec1, rec2, rec3]:
            if rec.get("format_recommande") == format_gagnant:
                qualites = [
                    r.get("qualite_recommandee", 85)
                    for r in [rec1, rec2, rec3]
                    if r.get("format_recommande") == format_gagnant
                ]
                rec["qualite_recommandee"] = int(sum(qualites) / len(qualites))
                rec["llm_source"]          = f"vote_majoritaire_{nb_votes}_sur_3"
                rec["statut"]              = "succes"
                return rec

        return self._recommandation_par_defaut(rapport_agent1, "defaut")

    # ------------------------------------------------------------------
    def _verifier_accord(self, rec1, rec2, rec3):
        formats = {
            rec1.get("format_recommande"),
            rec2.get("format_recommande"),
            rec3.get("format_recommande")
        }
        if len(formats) == 1:
            return "accord_total"
        elif len(formats) == 2:
            return "accord_partiel_2_sur_3"
        else:
            return "desaccord_total"

    # ------------------------------------------------------------------
    def _recommandation_par_defaut(self, rapport_agent1, source, erreur=""):
        categorie = rapport_agent1.get("categorie", "photos")
        score     = rapport_agent1.get("complexite", {}).get("score_complexite", 0.5)
        ocr       = rapport_agent1.get("ocr", {})

        if ocr.get("texte_detecte") and ocr.get("nb_mots", 0) > 20:
            format_rec, qualite = "PNG", 95
        elif categorie == "documents":
            format_rec, qualite = "PNG",  95
        elif categorie == "graphiques":
            format_rec, qualite = "PNG",  90
        elif score > 0.6:
            format_rec, qualite = "JPEG", 85
        else:
            format_rec, qualite = "WEBP", 80

        return {
            "format_recommande"       : format_rec,
            "qualite_recommandee"     : qualite,
            "taux_compression_estime" : 60,
            "justification"           : f"Defaut ({source})",
            "llm_source"              : source,
            "statut"                  : "defaut"
        }

    # ------------------------------------------------------------------
    def sauvegarder_recommandation(self, recommandation, chemin_sortie):
        os.makedirs(os.path.dirname(chemin_sortie), exist_ok=True)
        with open(chemin_sortie, "w", encoding="utf-8") as f:
            json.dump(recommandation, f, indent=4, ensure_ascii=False)
        print(f"Recommandation sauvegardee : {chemin_sortie}")