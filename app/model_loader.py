import joblib
import os

MODEL_PATH = os.path.join("models", "modelo_fakenews.pkl")

model = joblib.load(MODEL_PATH)

def get_model():
    return model
