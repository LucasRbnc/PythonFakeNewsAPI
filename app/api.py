from flask import Flask, request, jsonify
from app.model_loader import get_model

app = Flask(__name__)
model = get_model()

@app.route("/api/classificar-noticia", methods=["POST"])
def classificar():
    data = request.get_json()
    texto = data.get("texto", "")
    if not texto:
        return jsonify({"erro": "Texto não fornecido."}), 400

    pred = model.predict([texto])[0]
    prob = max(model.predict_proba([texto])[0])

    return jsonify({
        "classe": pred,
        "probabilidade": round(float(prob), 2)
    })

@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({"modelo": "Naive Bayes com TF-IDF", "status": "ativo"})
