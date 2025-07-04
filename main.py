from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import joblib
import uvicorn
import datetime

modelo = joblib.load("app/model/modelo_noticias.joblib")
app = FastAPI()

# Configuração do CORS para permitir requisições
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def palavras_mais_influentes(modelo_pipeline, texto, top_n=5):
    try:
        vectorizer = modelo_pipeline.named_steps['tfidfvectorizer']
        classifier = modelo_pipeline.named_steps['logisticregression']
    except Exception as e:
        print("Erro ao extrair vetorizador/classificador:", e)
        return []

    vetor = vectorizer.transform([texto])
    coef = classifier.coef_[0]
    indices_ativos = vetor.nonzero()[1]

    palavras_pesos = []
    for idx in indices_ativos:
        palavra = vectorizer.get_feature_names_out()[idx]
        peso = coef[idx]
        palavras_pesos.append((palavra, peso))

    palavras_ordenadas = sorted(palavras_pesos, key=lambda x: abs(x[1]), reverse=True)

    return [p[0] for p in palavras_ordenadas[:top_n]]

@app.post("/api/classificar-noticia", response_model=RespostaClassificacao)
def classificar_noticia(req: RequisicaoNoticia):
    print("📨 Requisição recebida:", req.texto)
    texto = req.texto
    probas = modelo.predict_proba([texto])[0]
    classe_idx = int(probas.argmax())
    probabilidade = float(probas[classe_idx])
    classe = "verdadeira" if classe_idx == 1 else "falsa"

    palavras_influentes = palavras_mais_influentes(modelo, texto, top_n=5)

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
