from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import shap
import os

# Initialisation de l'application FastAPI
app = FastAPI(
    title="ERP Club - AI Service",
    description="Microservice IA pour la prédiction globale et anatomique des blessures",
    version="3.1.0"
)

# ---------------------------------------------------------
# 1. CHARGEMENT ROBUSTE DES MODÈLES (Au démarrage)
# ---------------------------------------------------------
# BASE_DIR pointe vers la racine de ton projet (au-dessus du dossier 'app')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Modèle 1 : Risque Global (XGBoost) ---
MODEL_XGB_PATH = os.path.join(BASE_DIR, "ml_core", "artifacts", "injury_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "ml_core", "artifacts", "scaler.pkl")

try:
    xgb_model = joblib.load(MODEL_XGB_PATH)
    scaler = joblib.load(SCALER_PATH)
    explainer = shap.TreeExplainer(xgb_model)
    print(f"✅ [MODULE 1] Modèle XGBoost et Scaler chargés avec succès depuis : {MODEL_XGB_PATH}")
except Exception as e:
    xgb_model = None
    print(f"⚠️ [MODULE 1] Erreur lors du chargement de XGBoost : {e}")

# --- Modèle 2 : Cartographie des Zones (Random Forest / LightGBM) ---
model_zone_artifact = None

# Chemins explicites incluant le dossier "ml_core/models" que tu viens d'indiquer
possible_zone_paths = [
    os.path.join(BASE_DIR, "ml_core", "models", "injury_zone_model.joblib"), # Ton chemin exact !
    os.path.join(BASE_DIR, "ml_core", "artifacts", "injury_zone_model.joblib"),
    os.path.join(BASE_DIR, "models", "injury_zone_model.joblib"),
    os.path.join(BASE_DIR, "injury_zone_model.joblib")
]

for path in possible_zone_paths:
    if os.path.exists(path):
        try:
            model_zone_artifact = joblib.load(path)
            print(f"✅ [MODULE 2] Modèle 'Zone de Blessure' chargé avec succès depuis : {path}")
            break
        except Exception as e:
            print(f"⚠️ [MODULE 2] Erreur lors de la lecture du fichier {path} : {e}")

if not model_zone_artifact:
    print("⚠️ [MODULE 2] Fichier 'injury_zone_model.joblib' introuvable dans le projet. L'endpoint /predict-injury-zone renverra une erreur 500.")


# ---------------------------------------------------------
# 2. SCHÉMAS DE DONNÉES (Pydantic)
# ---------------------------------------------------------
# Input pour le Modèle 1 (XGBoost)
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
    # Ajout de valeurs par défaut pour corriger l'erreur 422 (Missing fields)
    sommeil_7d_mean: float = 7.0
    fatigue_7d_mean: float = 4.0
    douleurMusculaire_7d_mean: float = 3.0
    stress_7d_mean: float = 4.0
    model: str = "XGBoost (default)" # Gardé pour la compatibilité avec le frontend

# Ordre strict pour XGBoost
FEATURES_ORDER_XGB = [
    'totalLoad', 'sommeil', 'fatigue', 'douleurMusculaire', 'stress', 
    'acuteLoad', 'chronicLoad', 'ACWR', 
    'sommeil_7d_mean', 'fatigue_7d_mean', 'douleurMusculaire_7d_mean', 'stress_7d_mean'
]

# Input pour le Modèle 2 (Zones)
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

# ---------------------------------------------------------
# 3. ROUTES DE L'API
# ---------------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "ERP Club AI Service est en ligne 🟢"}

# ENDPOINT 1 : PRÉDICTION GLOBALE (XGBoost)
@app.post("/predict-injury")
def predict_injury_risk(data: PlayerFeatures):
    if not xgb_model:
        raise HTTPException(status_code=500, detail="Modèle XGBoost non chargé.")
        
    try:
        # 1. Structurer l'input dans l'ordre exact requis
        features_dict = {col: [getattr(data, col)] for col in FEATURES_ORDER_XGB}
        df_input = pd.DataFrame(features_dict)
        
        # 2. Standardisation
        df_scaled = pd.DataFrame(scaler.transform(df_input), columns=FEATURES_ORDER_XGB)
        
        # 3. Prédiction
        risk_prob = float(xgb_model.predict_proba(df_scaled)[0][1])
        
        # 4. SHAP Values
        shap_values = explainer.shap_values(df_scaled)
        if isinstance(shap_values, list):
            contributions = shap_values[1][0]
        else:
            contributions = shap_values[0] if len(shap_values.shape) > 1 else shap_values
            
        factors = []
        for feature_name, contrib in zip(FEATURES_ORDER_XGB, contributions):
            if abs(contrib) > 0.01:
                factors.append({
                    "feature": feature_name,
                    "contribution": round(float(contrib), 3),
                    "impact": "négatif" if contrib > 0 else "positif"
                })
        
        factors = sorted(factors, key=lambda x: abs(x["contribution"]), reverse=True)
        
        # 5. Définition logique
        if risk_prob > 0.70: risk_level = "Critique"
        elif risk_prob > 0.40: risk_level = "Modéré"
        else: risk_level = "Faible"
            
        return {
            "playerId": data.playerId,
            "riskScore": round(risk_prob, 2),
            "riskLevel": risk_level,
            "factors": factors
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction Global: {str(e)}")

# ENDPOINT 2 : CARTOGRAPHIE DES ZONES (Joblib Multi-classe)
@app.post("/predict-injury-zone")
def predict_injury_zone(data: ZonePredictionInput):
    if not model_zone_artifact:
        raise HTTPException(status_code=500, detail="Le modèle .joblib des zones n'a pas été trouvé au démarrage.")
    
    try:
        model_zone = model_zone_artifact.get('model')
        pos_encoder = model_zone_artifact.get('pos_mapping')
        foot_encoder = model_zone_artifact.get('foot_mapping')
        
        # Gestion dynamique des noms de clés du dictionnaire .joblib (V1 vs V2 du notebook)
        features = model_zone_artifact.get('feature_names', model_zone_artifact.get('features', []))
        zones = model_zone_artifact.get('model_classes', model_zone_artifact.get('target_classes', []))
        
        # 1. Encodage avec tolérance (si classe inconnue, on met 0 par défaut)
        pos_enc, foot_enc = 0, 0
        if pos_encoder:
            try: pos_enc = pos_encoder.transform([data.position])[0]
            except: pass
        if foot_encoder:
            try: foot_enc = foot_encoder.transform([data.foot])[0]
            except: pass
            
        # 2. Remplissage dynamique des variables
        input_dict = {
            'Age': data.age,
            'FIFA rating': data.fifa_rating,
            'acuteLoad': data.acuteLoad,
            'chronicLoad': data.chronicLoad,
            'ACWR': data.ACWR,
            'douleurMusculaire': data.douleurMusculaire,
            'souplesse': data.souplesse,
            'agilite': data.agilite,
            'Position_encoded': pos_enc,
            'Foot_encoded': foot_enc
        }
        
        # Construction du dictionnaire final respectant l'ordre exact de `features`
        final_input = {feat: input_dict.get(feat, 0) for feat in features}
        input_df = pd.DataFrame([final_input])[features] 
        
        # 3. Prédiction (predict_proba)
        probabilities = model_zone.predict_proba(input_df)[0]
        
        # 4. Formatage
        predictions = {zones[i]: float(probabilities[i]) for i in range(len(zones))}
        
        return {
            "playerId": data.playerId,
            "predictions": predictions
        }
    except Exception as e:
        # En cas d'erreur de dimension ou de scikit-learn, on remonte l'erreur exacte au front
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction Zone: {str(e)}")