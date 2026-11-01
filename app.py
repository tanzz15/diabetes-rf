import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import mlflow.pyfunc

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Diagnosis Diabetes RF + SMOTE + GWO")
templates = Jinja2Templates(directory="app/templates")

MODEL_SOURCE = os.getenv("MODEL_SOURCE", "mlflow")
MODEL_URI = os.getenv("MODEL_URI", "models:/rf_smote_gwo_diabetes/2")
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "models/rf_smote_gwo_model.pkl")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

# threshold hasil tuning eksperimen
# kalau belum tahu isi pastinya, sementara 0.5 dulu
BEST_THRESHOLD = float(os.getenv("BEST_THRESHOLD", "0.5"))

FEATURES = [
    "Glucose",
    "BMI",
    "Age",
    "DiabetesPedigreeFunction",
    "Pregnancies"
]


def load_model():
    if MODEL_SOURCE == "mlflow":
        if MLFLOW_TRACKING_URI:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # Load sebagai sklearn model supaya bisa pakai predict_proba()
        try:
            return mlflow.sklearn.load_model(MODEL_URI)
        except Exception:
            return mlflow.pyfunc.load_model(MODEL_URI)

    return joblib.load(LOCAL_MODEL_PATH)


model = load_model()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"result": None}
    )


@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    glucose: float = Form(...),
    bmi: float = Form(...),
    age: float = Form(...),
    diabetes_pedigree_function: float = Form(...),
    pregnancies: float = Form(...),
):
    input_data = pd.DataFrame([{
        "Glucose": glucose,
        "BMI": bmi,
        "Age": age,
        "DiabetesPedigreeFunction": diabetes_pedigree_function,
        "Pregnancies": pregnancies
    }])

    input_data = input_data[FEATURES]

    # Kalau model sklearn RandomForest, ini yang benar:
    # ambil probabilitas kelas 1 / diabetes
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_data)[0][1])
        threshold = BEST_THRESHOLD
        prediction = int(probability >= threshold)

    # Kalau ternyata yang keload masih custom pyfunc
    else:
        raw_result = model.predict(input_data)

        if isinstance(raw_result, pd.DataFrame):
            prediction = int(raw_result["prediction"].iloc[0])
            probability = float(raw_result["probability_diabetes"].iloc[0])
            threshold = float(raw_result["threshold"].iloc[0])
        else:
            prediction = int(raw_result[0])
            probability = 0.0
            threshold = BEST_THRESHOLD

    label = "Positif Diabetes" if prediction == 1 else "Negatif Diabetes"

    result = {
        "label": label,
        "probability": round(probability * 100, 2),
        "threshold": round(threshold, 2)
    }

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"result": result}
    )


@app.post("/reload")
def reload_model():
    global model
    model = load_model()
    return {
        "message": "Model berhasil direload",
        "model_source": MODEL_SOURCE,
        "model_uri": MODEL_URI
    }