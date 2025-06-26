import os
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Baixar e copiar os datasets
data_dir = "data/raw"
os.makedirs(data_dir, exist_ok=True)

path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")

# Carregar os dados
fake = pd.read_csv(os.path.join(path, "Fake.csv"))
true = pd.read_csv(os.path.join(path, "True.csv"))

fake["label"] = "falsa"
true["label"] = "verdadeira"

df = pd.concat([fake, true])
df = df[["text", "label"]].dropna()

# Treinar modelo
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.7)),
    ("clf", MultinomialNB())
])

model.fit(X_train, y_train)

print("Acurácia:", model.score(X_test, y_test))

# Salvar modelo
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/modelo_fakenews.pkl")