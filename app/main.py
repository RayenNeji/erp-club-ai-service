from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import shap
import os

# Initialisation de l'application FastAPI
app = FastAPI(
    title="ERP Club - AI Service",
    description="Microservice IA : Risque Global, Zones Anatomiques, et Survie de Rechute",
    version="4.0.0"
)

# ---------------------------------------------------------
# 1. CHARGEMENT ROBUSTE DES MODÈLES (Au démarrage)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- MODULE 1 : Risque Global (XGBoost) ---
MODEL_XGB_PATH = os.path.join(BASE_DIR, "ml_core", "artifacts", "injury_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "ml_core", "artifacts", "scaler.pkl")

try:
    xgb_model = joblib.load(MODEL_XGB_PATH)
    scaler = joblib.load(SCALER_PATH)
    explainer = shap.TreeExplainer(xgb_model)
    print(f"✅ [MODULE 1] Modèle XGBoost chargé avec succès.")
except Exception as e:
    xgb_model = None
    print(f"⚠️ [MODULE 1] Erreur XGBoost : {e}")

# --- MODULE 2 : Cartographie des Zones (Random Forest / LightGBM) ---
model_zone_artifact = None
possible_zone_paths = [
    os.path.join(BASE_DIR, "ml_core", "models", "injury_zone_model.joblib"),
    os.path.join(BASE_DIR, "models", "injury_zone_model.joblib"),
    os.path.join(BASE_DIR, "injury_zone_model.joblib")
]

for path in possible_zone_paths:
    if os.path.exists(path):
        try:
            model_zone_artifact = joblib.load(path)
            print(f"✅ [MODULE 2] Modèle 'Zone de Blessure' chargé avec succès.")
            break
        except Exception: pass
if not model_zone_artifact:
    print("⚠️ [MODULE 2] Fichier 'injury_zone_model.joblib' introuvable.")

# --- MODULE 3 : Analyse de Survie (Cox Proportional Hazards) ---
model_survival_artifact = None
possible_survival_paths = [
    os.path.join(BASE_DIR, "ml_core", "models", "relapse_survival_model.joblib"),
    os.path.join(BASE_DIR, "models", "relapse_survival_model.joblib"),
    os.path.join(BASE_DIR, "relapse_survival_model.joblib")
]

for path in possible_survival_paths:
    if os.path.exists(path):
        try:
            model_survival_artifact = joblib.load(path)
            print(f"✅ [MODULE 3] Modèle 'Analyse de Survie' chargé avec succès.")
            break
        except Exception as e: 
            print(f"⚠️ [MODULE 3] Erreur de lecture : {e}")
if not model_survival_artifact:
    print("⚠️ [MODULE 3] Fichier 'relapse_survival_model.joblib' introuvable.")


# ---------------------------------------------------------
# 2. SCHÉMAS DE DONNÉES (Pydantic)
# ---------------------------------------------------------

class PlayerFeatures(BaseModel):
    playerId: int
    totalLoad: float
    sommeil: float
    fatigue: float
    douleurMusculaire: float
    stress: float
    acuteLoad: float
    chronicLoad: float
    ACWR: float
    sommeil_7d_mean: float = 7.0
    fatigue_7d_mean: float = 4.0
    douleurMusculaire_7d_mean: float = 3.0
    stress_7d_mean: float = 4.0
    model: str = "XGBoost (default)"

FEATURES_ORDER_XGB = [
    'totalLoad', 'sommeil', 'fatigue', 'douleurMusculaire', 'stress', 
    'acuteLoad', 'chronicLoad', 'ACWR', 
    'sommeil_7d_mean', 'fatigue_7d_mean', 'douleurMusculaire_7d_mean', 'stress_7d_mean'
]

class ZonePredictionInput(BaseModel):
    playerId: int
    position: str
    foot: str
    age: int
    fifa_rating: int
    acuteLoad: float
    chronicLoad: float
    ACWR: float
    douleurMusculaire: float
    souplesse: float
    agilite: float

# Input pour le Modèle 3 (Survie)
class RelapseSurvivalInput(BaseModel):
    playerId: int
    recovery_score: float
    sleep_quality: float
    stress_level: float
    fatigue_index: float
    physio_adherence: float
    post_recovery_ACWR: float

# ---------------------------------------------------------
# 3. ROUTES DE L'API
# ---------------------------------------------------------

@app.get("/")
def read_root():
    return {"message": "ERP Club AI Service est en ligne 🟢"}

@app.post("/predict-injury")
def predict_injury_risk(data: PlayerFeatures):
    if not xgb_model:
        raise HTTPException(status_code=500, detail="Modèle XGBoost non chargé.")
    try:
        features_dict = {col: [getattr(data, col)] for col in FEATURES_ORDER_XGB}
        df_input = pd.DataFrame(features_dict)
        df_scaled = pd.DataFrame(scaler.transform(df_input), columns=FEATURES_ORDER_XGB)
        risk_prob = float(xgb_model.predict_proba(df_scaled)[0][1])
        
        shap_values = explainer.shap_values(df_scaled)
        contributions = shap_values[1][0] if isinstance(shap_values, list) else (shap_values[0] if len(shap_values.shape) > 1 else shap_values)
            
        factors = []
        for feature_name, contrib in zip(FEATURES_ORDER_XGB, contributions):
            if abs(contrib) > 0.01:
                factors.append({"feature": feature_name, "contribution": round(float(contrib), 3), "impact": "négatif" if contrib > 0 else "positif"})
        factors = sorted(factors, key=lambda x: abs(x["contribution"]), reverse=True)
        
        risk_level = "Critique" if risk_prob > 0.70 else ("Modéré" if risk_prob > 0.40 else "Faible")
        return {"playerId": data.playerId, "riskScore": round(risk_prob, 2), "riskLevel": risk_level, "factors": factors}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-injury-zone")
def predict_injury_zone(data: ZonePredictionInput):
    if not model_zone_artifact:
        raise HTTPException(status_code=500, detail="Le modèle des zones est introuvable.")
    try:
        model_zone = model_zone_artifact.get('model')
        pos_encoder, foot_encoder = model_zone_artifact.get('pos_mapping'), model_zone_artifact.get('foot_mapping')
        features = model_zone_artifact.get('feature_names', model_zone_artifact.get('features', []))
        zones = model_zone_artifact.get('model_classes', model_zone_artifact.get('target_classes', []))
        
        pos_enc = pos_encoder.transform([data.position])[0] if pos_encoder and data.position in pos_encoder.classes_ else 0
        foot_enc = foot_encoder.transform([data.foot])[0] if foot_encoder and data.foot in foot_encoder.classes_ else 0
            
        input_dict = {
            'Age': data.age, 'FIFA rating': data.fifa_rating, 'acuteLoad': data.acuteLoad,
            'chronicLoad': data.chronicLoad, 'ACWR': data.ACWR, 'douleurMusculaire': data.douleurMusculaire,
            'souplesse': data.souplesse, 'agilite': data.agilite, 'Position_encoded': pos_enc, 'Foot_encoded': foot_enc
        }
        
        final_input = {feat: input_dict.get(feat, 0) for feat in features}
        input_df = pd.DataFrame([final_input])[features] 
        probabilities = model_zone.predict_proba(input_df)[0]
        predictions = {zones[i]: float(probabilities[i]) for i in range(len(zones))}
        return {"playerId": data.playerId, "predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# NOUVEAU : ENDPOINT 3 - SURVIE DES RECHUTES (Cox PH)
@app.post("/predict-relapse")
def predict_relapse_risk(data: RelapseSurvivalInput):
    if not model_survival_artifact:
        raise HTTPException(status_code=500, detail="Le modèle de Survie (Cox) est introuvable.")
    
    try:
        model_surv = model_survival_artifact['model']
        scaler_surv = model_survival_artifact['scaler']
        features_surv = model_survival_artifact['features']
        
        # 1. Construction du DataFrame de la requête
        input_dict = {
            'recovery_score': data.recovery_score,
            'sleep_quality': data.sleep_quality,
            'stress_level': data.stress_level,
            'fatigue_index': data.fatigue_index,
            'physio_adherence': data.physio_adherence,
            'post_recovery_ACWR': data.post_recovery_ACWR
        }
        
        input_df = pd.DataFrame([input_dict])[features_surv]
        
        # 2. Standardisation des variables pour correspondre à l'entraînement
        input_scaled = pd.DataFrame(scaler_surv.transform(input_df), columns=features_surv)
        
        # 3. Prédiction de la fonction de survie (Lifelines)
        surv_func = model_surv.predict_survival_function(input_scaled)
        
        # 4. Formatage de la courbe (Jour vs Probabilité) pour le front-end
        timeline = surv_func.index.tolist()
        probabilities = surv_func.iloc[:, 0].tolist()
        
        curve = [{"day": int(t), "probability": float(p)} for t, p in zip(timeline, probabilities)]
        
        return {
            "playerId": data.playerId,
            "c_index": model_survival_artifact.get('c_index', 0.96),
            "survival_curve": curve
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction Survie: {str(e)}")