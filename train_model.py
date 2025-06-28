import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

df = pd.read_csv("data/pre-processed.csv")
df["label"] = df["label"].str.lower().str.strip()
df = df.dropna(subset=["preprocessed_news", "label"])
df = df[df["label"].isin(["fake", "true"])]

# Mapeamento correto agora
y = df["label"].map({"fake": 0, "true": 1})
X = df["preprocessed_news"]

# Checagem final
if X.empty or y.isnull().any():
    raise ValueError("Erro: coluna 'label' ainda contém valores inválidos ou ausentes.")

# Treinamento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("clf", LogisticRegression())
])

pipeline.fit(X_train, y_train)

# Salvar modelo
joblib.dump(pipeline, "app/model/modelo_noticias.joblib")
print("✅ Modelo treinado e salvo com sucesso.")
