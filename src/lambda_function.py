import json
import os
import logging
import joblib
import numpy as np
import lightgbm as lgb
from datetime import datetime

from features import build_features

# =========================
# LOGGING (CloudWatch)
# =========================
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "modelo")

MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_model.txt")  # 👈 usando TXT
STATS_PATH = os.path.join(MODEL_DIR, "stats.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "features_list.pkl")

# =========================
# CARGA EN FRÍO (una vez)
# =========================
try:
    model = lgb.Booster(model_file=MODEL_PATH)
    stats = joblib.load(STATS_PATH)
    features_list = joblib.load(FEATURES_PATH)

    logger.info("✅ Modelo y artefactos cargados correctamente")

except Exception as e:
    logger.error(f"❌ Error cargando modelo: {e}")
    raise e


# =========================
# VALIDACIÓN BÁSICA
# =========================
REQUIRED_FIELDS = [
    "danceability", "energy", "loudness", "valence",
    "speechiness", "instrumentalness", "liveness",
    "acousticness", "duration_ms", "key", "mode",
    "explicit", "track_name", "artists",
    "album_name", "track_genre"
]


def validate_input(data):
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        raise ValueError(f"Missing fields: {missing}")


# =========================
# PREDICCIÓN
# =========================
def predict(data):
    df = build_features(data, stats, features_list)

    pred = model.predict(df)

    if isinstance(pred, (list, np.ndarray)):
        pred = pred[0]

    return float(np.clip(pred, 0, 100))


# =========================
# HANDLER AWS
# =========================
def lambda_handler(event, context):
    try:
        # Manejo del body (API Gateway manda string)
        body = event.get("body", event)

        if isinstance(body, str):
            body = json.loads(body)

        # Predicción
        prediction = predict(body) * 100

        # Interpretación
        if prediction > 70:
            interpretacion = "Alta popularidad (>70)"
        elif prediction > 30:
            interpretacion = "Media popularidad (30-70)"
        else:
            interpretacion = "Baja popularidad (<30)"

        # Respuesta enriquecida
        response = {
            "cancion": body.get("track_name", "Desconocida"),
            "artista": body.get("artists", "Desconocido"),
            "popularidad_predicha": round(prediction, 2),
            "interpretacion": interpretacion,
            "metadata": {
                "modelo": "LightGBM",
                "version": "1.0",
                "timestamp": datetime.utcnow().isoformat()
            }
        }

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps(response)
        }

    except Exception as e:
        return {
            "statusCode": 400,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({
                "error": str(e),
                "tipo": "Bad Request"
            })
        }
