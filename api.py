import os
import sys
import pathlib
import json
import time
import glob
from datetime import datetime
import os
# Calcule automatiquement la racine du projet, peu importe l'ordinateur
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


sys.path.insert(0, str(pathlib.Path(__file__).parent))

from flask import Flask, request, jsonify
from agents.agent_analyseur   import AgentAnalyseur
from agents.agent_classifier  import AgentClassifier
from agents.agent_compresseur import AgentCompresseur
from agents.agent_evaluateur  import AgentEvaluateur
from agents.agent_rapporteur  import AgentRapporteur

app    = Flask(__name__)
agent1 = AgentAnalyseur()
agent2 = AgentClassifier()
agent3 = AgentCompresseur()
agent4 = AgentEvaluateur()
agent5 = AgentRapporteur()

print("=" * 50)
print("  API Agents - Compression Intelligente")
print("  Agent 1 : AgentAnalyseur     OK")
print("  Agent 2 : AgentClassifier    OK")
print("  Agent 3 : AgentCompresseur   OK")
print("  Agent 4 : AgentEvaluateur    OK")
print("  Agent 5 : AgentRapporteur    OK")
print("=" * 50)

# ============================================================
# UTILITAIRE : Retry automatique
# ============================================================
def appeler_avec_retry(fonction, max_tentatives=3, delai=2, **kwargs):
    for tentative in range(1, max_tentatives + 1):
        try:
            print(f"    Tentative {tentative}/{max_tentatives}...")
            resultat = fonction(**kwargs)
            if resultat and resultat.get("statut") not in ["erreur"]:
                return resultat
            print(f"    Tentative {tentative} echouee")
        except Exception as e:
            print(f"    Tentative {tentative} exception : {e}")
        if tentative < max_tentatives:
            time.sleep(delai)
    print(f"    Tous les {max_tentatives} essais ont echoue")
    return None


# ============================================================
# UTILITAIRE : Pipeline avec Backtracking
# ============================================================
def pipeline_avec_backtracking(chemin_image, categorie, dossier_sortie,
                                max_backtrack=2, seuil_ssim=0.80):
    logs              = []
    rapport_compression = None
    rapport_evaluation  = None

    # Agent 1
    rapport = appeler_avec_retry(
        agent1.analyser, max_tentatives=3,
        chemin_image=chemin_image
    )
    if not rapport:
        return {"statut": "erreur", "message": "Agent 1 echoue"}
    rapport["categorie"] = categorie
    logs.append("Agent 1 : OK")

    # Agent 2
    recommandation = appeler_avec_retry(
        agent2.classifier, max_tentatives=3, delai=3,
        rapport_agent1=rapport
    )
    if not recommandation:
        return {"statut": "erreur", "message": "Agent 2 echoue"}
    logs.append(f"Agent 2 : OK → {recommandation.get('format_recommande')} "
                f"q={recommandation.get('qualite_recommandee')}%")

    # Agents 3+4 avec Backtracking
    qualite_actuelle = recommandation.get("qualite_recommandee", 85)
    meilleure_ssim   = 0
    nb_backtrack     = 0

    for tentative_bt in range(max_backtrack + 1):
        if tentative_bt > 0:
            qualite_actuelle = min(qualite_actuelle + 10, 95)
            recommandation["qualite_recommandee"] = qualite_actuelle
            print(f"  BACKTRACKING {tentative_bt}/{max_backtrack} "
                  f"→ qualite : q={qualite_actuelle}%")
            logs.append(f"Backtracking {tentative_bt} : q={qualite_actuelle}%")
            nb_backtrack += 1

        rapport_compression = appeler_avec_retry(
            agent3.compresser, max_tentatives=2,
            chemin_image=chemin_image,
            recommandation=recommandation,
            dossier_sortie=dossier_sortie
        )
        if not rapport_compression:
            logs.append("Agent 3 : ECHEC")
            break

        rapport_evaluation = appeler_avec_retry(
            agent4.evaluer, max_tentatives=2,
            chemin_originale=chemin_image,
            rapport_compression=rapport_compression
        )
        if not rapport_evaluation:
            logs.append("Agent 4 : ECHEC")
            break

        meilleure      = rapport_evaluation.get("meilleure_compression", {})
        ssim_obtenu    = meilleure.get("ssim", 0)
        meilleure_ssim = ssim_obtenu
        logs.append(f"Tentative {tentative_bt+1} : SSIM={ssim_obtenu} | "
                    f"PSNR={meilleure.get('psnr_db')} dB")
        print(f"  SSIM : {ssim_obtenu} "
              f"({'OK' if ssim_obtenu >= seuil_ssim else 'insuffisant'})")

        if ssim_obtenu >= seuil_ssim:
            print(f"  Qualite suffisante — arret backtracking")
            break
        if tentative_bt >= max_backtrack:
            logs.append(f"Max backtracking atteint — SSIM : {ssim_obtenu}")

    # Agent 5
    rapport_final = agent5.generer_rapport(
        rapport_analyse     = rapport,
        recommandation      = recommandation,
        rapport_compression = rapport_compression,
        rapport_evaluation  = rapport_evaluation
    )
    logs.append("Agent 5 : OK")

    rapport_final["backtracking"] = {
        "nb_backtracking": nb_backtrack,
        "ssim_final"     : meilleure_ssim,
        "qualite_finale" : qualite_actuelle,
        "seuil_ssim"     : seuil_ssim,
        "logs"           : logs
    }
    return rapport_final


# ============================================================
# ROUTES
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "statut" : "ok",
        "message": "API Agents operationnelle",
        "routes" : [
            "GET  /health",
            "POST /analyser",
            "POST /classifier",
            "POST /compresser",
            "POST /evaluer",
            "POST /rapport",
            "POST /pipeline",
            "POST /batch"
        ]
    }), 200


@app.route("/analyser", methods=["POST"])
def analyser():
    try:
        data         = request.json
        chemin_image = data.get("chemin_image", "")
        if not chemin_image:
            return jsonify({"statut": "erreur", "message": "chemin_image manquant"}), 400
        if not os.path.exists(chemin_image):
            return jsonify({"statut": "erreur", "message": f"Image non trouvee"}), 404
        rapport = agent1.analyser(chemin_image)
        return jsonify(rapport), 200
    except Exception as e:
        return jsonify({"statut": "erreur", "message": str(e)}), 500


@app.route("/classifier", methods=["POST"])
def classifier():
    try:
        data      = request.json
        rapport   = data.get("rapport", {})
        categorie = data.get("categorie", "photos")
        if not rapport:
            return jsonify({"statut": "erreur", "message": "rapport manquant"}), 400
        rapport["categorie"] = categorie
        recommandation       = agent2.classifier(rapport)
        return jsonify({
            "format_recommande"       : recommandation.get("format_recommande", "JPEG"),
            "qualite_recommandee"     : recommandation.get("qualite_recommandee", 85),
            "taux_compression_estime" : recommandation.get("taux_compression_estime", 70),
            "justification"           : recommandation.get("justification", ""),
            "priorite"                : recommandation.get("priorite", ""),
            "multi_llm"               : recommandation.get("multi_llm", {}),
            "statut"                  : "succes"
        }), 200
    except Exception as e:
        return jsonify({"statut": "erreur", "message": str(e)}), 500


@app.route("/compresser", methods=["POST"])
def compresser():
    try:
        data = request.json if request.is_json else request.form.to_dict()
        
        # Support pour variables directes ou imbriquées (typique dans n8n)
        if "recommandation" in data and isinstance(data["recommandation"], dict):
            rec = data["recommandation"]
            format_recommande = rec.get("format_recommande", data.get("format_recommande", "JPEG"))
            qualite_recommandee = int(rec.get("qualite_recommandee", data.get("qualite_recommandee", 85)))
        else:
            format_recommande   = data.get("format_recommande", "JPEG")
            qualite_recommandee = int(data.get("qualite_recommandee", 85))
            
        chemin_image   = data.get("chemin_image", "")
        categorie      = data.get("categorie", "photos")
        dossier_sortie = data.get("dossier_sortie")
        if not dossier_sortie:
            dossier_sortie = os.path.join(BASE_DIR, "results", categorie)
        os.makedirs(dossier_sortie, exist_ok=True)
        if not chemin_image:
            return jsonify({"statut": "erreur", "message": "chemin_image manquant"}), 400
        rapport = agent3.compresser(
            chemin_image   = chemin_image,
            recommandation = {"format_recommande": format_recommande,
                              "qualite_recommandee": qualite_recommandee},
            dossier_sortie = dossier_sortie
        )
        return jsonify(rapport), 200
    except Exception as e:
        return jsonify({"statut": "erreur", "message": str(e)}), 500


@app.route("/evaluer", methods=["POST"])
def evaluer():
    try:
        data                = request.json
        chemin_originale    = data.get("chemin_originale", "")
        rapport_compression = data.get("rapport_compression", {})
        if isinstance(rapport_compression, str):
            rapport_compression = json.loads(rapport_compression)
        if not chemin_originale:
            return jsonify({"statut": "erreur", "message": "chemin_originale manquant"}), 400
        rapport = agent4.evaluer(
            chemin_originale    = chemin_originale,
            rapport_compression = rapport_compression
        )
        return jsonify(rapport), 200
    except Exception as e:
        return jsonify({"statut": "erreur", "message": str(e)}), 500


@app.route("/rapport", methods=["POST"])
def rapport():
    try:
        data = request.json if request.is_json else request.form.to_dict()

        # Fonction bouclier : parse le JSON sans jamais faire planter Flask
        def safe_loads(val):
            if isinstance(val, dict):
                return val
            if isinstance(val, str):
                val = val.strip()
                if not val:
                    return {}
                try:
                    return json.loads(val)
                except:
                    return {}
            return {}

        rapport_analyse     = safe_loads(data.get("rapport_analyse"))
        recommandation      = safe_loads(data.get("recommandation"))
        rapport_compression = safe_loads(data.get("rapport_compression"))
        rapport_evaluation  = safe_loads(data.get("rapport_evaluation"))

        if not rapport_analyse:
            return jsonify({"statut": "erreur", "message": "rapport_analyse manquant"}), 400

        rapport_final = agent5.generer_rapport(
            rapport_analyse     = rapport_analyse,
            recommandation      = recommandation,
            rapport_compression = rapport_compression,
            rapport_evaluation  = rapport_evaluation
        )
        return jsonify(rapport_final), 200

    except Exception as e:
        return jsonify({"statut": "erreur", "message": str(e)}), 500


@app.route("/pipeline", methods=["POST"])
def pipeline():
    try:
        data           = request.json
        chemin_image   = data.get("chemin_image", "")
        categorie      = data.get("categorie", "photos")
        dossier_sortie = data.get("dossier_sortie")
        if not dossier_sortie:
            dossier_sortie = os.path.join(BASE_DIR, "results", categorie)
        os.makedirs(dossier_sortie, exist_ok=True)
        if not chemin_image:
            return jsonify({"statut": "erreur", "message": "chemin_image manquant"}), 400
        if not os.path.exists(chemin_image):
            return jsonify({"statut": "erreur", "message": "Image non trouvee"}), 404

        resultat = pipeline_avec_backtracking(
            chemin_image   = chemin_image,
            categorie      = categorie,
            dossier_sortie = dossier_sortie
        )
        return jsonify(resultat), 200
    except Exception as e:
        return jsonify({"statut": "erreur", "message": str(e)}), 500


@app.route("/batch", methods=["POST"])
def batch():
    try:
        data         = request.json
        base_dataset = data.get("base_dataset", os.path.join(BASE_DIR, "dataset"))
        base_results = data.get("base_results", os.path.join(BASE_DIR, "results"))
        categories   = ["photos", "documents", "graphiques", "screenshots"]
        extensions   = ["*.jpg", "*.jpeg", "*.png", "*.tif",
                        "*.tiff", "*.bmp", "*.webp"]

        resume = {
            "date_batch"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total"        : 0,
            "succes"       : 0,
            "erreurs"      : 0,
            "backtrackings": 0,
            "resultats"    : []
        }

        for categorie in categories:
            dossier_images  = os.path.join(base_dataset, categorie)
            dossier_results = os.path.join(base_results,  categorie)
            os.makedirs(dossier_results, exist_ok=True)
            if not os.path.exists(dossier_images):
                continue

            images = []
            for ext in extensions:
                images += glob.glob(os.path.join(dossier_images, ext))
            print(f"\n📁 {categorie.upper()} : {len(images)} images")

            for chemin_image in images:
                nom = os.path.basename(chemin_image)
                resume["total"] += 1
                print(f"\n  {nom}")
                try:
                    resultat = pipeline_avec_backtracking(
                        chemin_image   = chemin_image,
                        categorie      = categorie,
                        dossier_sortie = dossier_results
                    )
                    nom_base = os.path.splitext(nom)[0]
                    agent5.sauvegarder_rapport(
                        resultat,
                        os.path.join(dossier_results, f"{nom_base}_rapport.json")
                    )
                    bt = resultat.get("backtracking", {})
                    resume["succes"] += 1
                    if bt.get("nb_backtracking", 0) > 0:
                        resume["backtrackings"] += 1
                    resume["resultats"].append({
                        "image"       : nom,
                        "categorie"   : categorie,
                        "ssim"        : bt.get("ssim_final"),
                        "nb_backtrack": bt.get("nb_backtracking", 0),
                        "statut"      : "succes"
                    })
                    print(f"  OK — SSIM={bt.get('ssim_final')} "
                          f"backtracking={bt.get('nb_backtracking', 0)}")
                except Exception as e:
                    resume["erreurs"] += 1
                    resume["resultats"].append({
                        "image"    : nom,
                        "categorie": categorie,
                        "statut"   : f"erreur: {str(e)[:50]}"
                    })
                    print(f"  Erreur : {str(e)[:80]}")

        chemin_resume = os.path.join(base_results, "resume_batch.json")
        with open(chemin_resume, "w", encoding="utf-8") as f:
            json.dump(resume, f, indent=4, ensure_ascii=False)

        print(f"\nBATCH TERMINE : {resume['succes']}/{resume['total']} succes")
        return jsonify(resume), 200

    except Exception as e:
        return jsonify({"statut": "erreur", "message": str(e)}), 500


# ============================================================
if __name__ == "__main__":
    print("\n Demarrage API sur http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)