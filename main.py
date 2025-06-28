from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Dict
import joblib
import uvicorn
import datetime

modelo = joblib.load("app/model/modelo_noticias.joblib")
app = FastAPI()

historico: List[Dict] = []

class RequisicaoNoticia(BaseModel):
    texto: str

class RespostaClassificacao(BaseModel):
    classe: str
    probabilidade: float
    palavras_influentes: List[str]

print("🟡 Carregando modelo...")
modelo = joblib.load("app/model/modelo_noticias.joblib")
print("🟢 Modelo carregado com sucesso!")


@app.post("/api/classificar-noticia", response_model=RespostaClassificacao)
def classificar_noticia(req: RequisicaoNoticia):
    print("📨 Requisição recebida:", req.texto)
    texto = req.texto
    probas = modelo.predict_proba([texto])[0]
    classe_idx = int(probas.argmax())
    probabilidade = float(probas[classe_idx])
    classe = "verdadeira" if classe_idx == 1 else "falsa"

    palavras_chave = ["escândalo", "urgente", "exclusivo"]
    palavras_influentes = [p for p in palavras_chave if p in texto.lower()]

    resultado = {
        "classe": classe,
        "probabilidade": round(probabilidade, 2),
        "palavras_influentes": palavras_influentes
    }

    historico.append({
        "texto": texto,
        "classe": classe,
        "probabilidade": probabilidade,
        "data": str(datetime.datetime.now())
    })

    print("✅ Classificação feita:", resultado)
    return resultado

@app.get("/api/historico")
def get_historico():
    return historico

@app.get("/api/status")
def get_status():
    return {
        "modelo_carregado": True,
        "total_classificacoes": len(historico)
    }

# Executar com: uvicorn main:app --reload
