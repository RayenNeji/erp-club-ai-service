from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import shap
import os

# Initialisation de l'application FastAPI
app = FastAPI(
    title="ERP Club - AI Service",
    description="Microservice IA pour la prédiction des blessures",
    version="1.0.0"
)

# ---------------------------------------------------------
# 1. CHARGEMENT DES MODÈLES (Au démarrage du serveur)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ml_core", "artifacts", "injury_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "ml_core", "artifacts", "scaler.pkl")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    # Initialisation de l'explainer SHAP
    explainer = shap.TreeExplainer(model)
    print("✅ Modèle XGBoost et Scaler chargés avec succès.")
except Exception as e:
    print(f"⚠️ Erreur lors du chargement des modèles : {e}")

# ---------------------------------------------------------
# 2. SCHÉMAS DE DONNÉES (Strictement aligné avec le Notebook)
# ---------------------------------------------------------
class PlayerFeatures(BaseModel):
    playerId: int
    # Valeurs du jour même
    totalLoad: float
    sommeil: float
    fatigue: float
    douleurMusculaire: float
    stress: float
    # Valeurs calculées (par NestJS/Prisma sur les fenêtres glissantes)
    acuteLoad: float
    chronicLoad: float
    ACWR: float
    sommeil_7d_mean: float
    fatigue_7d_mean: float
    douleurMusculaire_7d_mean: float
    stress_7d_mean: float

# L'ordre exact des colonnes requis par ton scaler et modèle XGBoost
FEATURES_ORDER = [
    'totalLoad', 'sommeil', 'fatigue', 'douleurMusculaire', 'stress', 
    'acuteLoad', 'chronicLoad', 'ACWR', 
    'sommeil_7d_mean', 'fatigue_7d_mean', 'douleurMusculaire_7d_mean', 'stress_7d_mean'
]

# ---------------------------------------------------------
# 3. ROUTES DE L'API
# ---------------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "ERP Club AI Service est en ligne 🟢"}

@app.post("/predict-injury")
def predict_injury_risk(data: PlayerFeatures):
    try:
        # 1. Structurer l'input dans l'ordre exact requis par le modèle
        features_dict = {col: [getattr(data, col)] for col in FEATURES_ORDER}
        df_input = pd.DataFrame(features_dict)
        
        # 2. Appliquer la standardisation (StandardScaler) sur le DataFrame
        df_scaled = pd.DataFrame(scaler.transform(df_input), columns=FEATURES_ORDER)
        
        # 3. Prédiction de la probabilité de blessure
        risk_prob = float(model.predict_proba(df_scaled)[0][1])
        
        # 4. Explicabilité : Calcul des contributions SHAP
        shap_values = explainer.shap_values(df_scaled)
        
        # Extraction des contributions selon la dimension de la matrice SHAP
        if isinstance(shap_values, list):
            contributions = shap_values[1][0]
        else:
            contributions = shap_values[0] if len(shap_values.shape) > 1 else shap_values
            
        factors = []
        for feature_name, contrib in zip(FEATURES_ORDER, contributions):
            if abs(contrib) > 0.01:  # Seuil pour filtrer les variables les plus impactantes
                factors.append({
                    "feature": feature_name,
                    "contribution": round(float(contrib), 3),
                    "impact": "négatif (augmente le risque)" if contrib > 0 else "positif (réduit le risque)"
                })
        
        # Trier du plus critique au moins critique
        factors = sorted(factors, key=lambda x: abs(x["contribution"]), reverse=True)
        
        # 5. Définition logique du niveau de risque
        if risk_prob > 0.70:
            risk_level = "Critique"
        elif risk_prob > 0.40:
            risk_level = "Modéré"
        else:
            risk_level = "Faible"
            
        return {
            "playerId": data.playerId,
            "riskScore": round(risk_prob, 2),
            "riskLevel": risk_level,
            "factors": factors
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))