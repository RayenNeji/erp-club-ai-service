import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import time

# 1. Configuration initiale
st.set_page_config(
    page_title="ERP Club AI - Service de Prédiction",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CSS Global unifié
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(145deg, #f6f9fc 0%, #e9f2f9 100%); }
    
    .header-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.2rem 2rem; border-radius: 1.5rem; margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15); color: white;
        display: flex; align-items: center; gap: 1.5rem;
    }
    .header-container h1 { font-weight: 800; font-size: 2rem; margin: 0; }
    .logo {
        font-size: 2.8rem; background: rgba(255,255,255,0.15); width: 3.8rem; height: 3.8rem;
        display: flex; align-items: center; justify-content: center; border-radius: 50%;
    }

    .card {
        background: rgba(255, 255, 255, 0.85); backdrop-filter: blur(8px);
        border-radius: 1.25rem; padding: 1.5rem 1.8rem; margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.04); border: 1px solid rgba(255, 255, 255, 0.5);
    }
    .card-title { font-weight: 700; font-size: 1.1rem; color: #1e3c72; border-bottom: 2px solid rgba(30, 60, 114, 0.1); padding-bottom: 0.6rem; margin-bottom: 1rem;}
    
    .metric-card { background: white; border-radius: 1rem; padding: 0.8rem; text-align: center; border: 1px solid rgba(30,60,114,0.08); }
    .metric-card .label { font-size: 0.8rem; color: #6b7a8f; text-transform: uppercase; }
    .metric-card .value { font-size: 1.6rem; font-weight: 800; color: #1e3c72; }
    
    .danger-alert {
        background: #fef2f2; border-left: 6px solid #ef4444; border-radius: 8px;
        padding: 20px; color: #991b1b; margin: 20px 0;
    }
    
    /* CSS pour la Heatmap Anatomique */
    .body-part {
        border: 1px solid #cbd5e1;
        display: flex; justify-content: center; align-items: center;
        font-size: 0.6rem; font-weight: bold; color: #334155;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# MENU DE NAVIGATION (SIDEBAR)
# ==========================================
st.sidebar.markdown("## 🧭 Navigation")
page = st.sidebar.radio(
    "Modules d'Intelligence Artificielle :",
    ["🩺 Risque de Blessure (Global)", "🗺️ Cartographie Anatomique (Zones)"]
)
st.sidebar.markdown("---")

API_URL_GLOBAL = "http://localhost:8000/predict-injury"
API_URL_ZONE = "http://localhost:8000/predict-injury-zone"

# ==========================================
# PAGE 1 : RISQUE GLOBAL (Modèle 1)
# ==========================================
if page == "🩺 Risque de Blessure (Global)":
    st.markdown("""
    <div class="header-container">
        <div class="logo">🩺</div>
        <div>
            <h1>ERP Club AI - Prédiction Globale</h1>
            <span style="opacity: 0.9;">Classification Binaire (Blessure imminente : Oui/Non)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚙️ Modèle Actif")
        model_choice = st.selectbox("Algorithme", ["XGBoost", "LightGBM", "Random Forest"])

    col1, col2 = st.columns([1.2, 1], gap="large")

    with col1:
        st.markdown('<div class="card"><div class="card-title">📋 Variables Cliniques et GPS</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            sommeil = st.slider("💤 Sommeil", 1.0, 10.0, 7.5)
            fatigue = st.slider("⚡ Fatigue", 1.0, 10.0, 4.0)
            acuteLoad = st.number_input("Acute Load", value=5950, step=100)
        with c2:
            douleur = st.slider("🦵 Douleurs", 1.0, 10.0, 3.0)
            stress = st.slider("🧘 Stress", 1.0, 10.0, 4.5)
            chronicLoad = st.number_input("Chronic Load", value=5100, step=100)
            
        totalLoad = st.number_input("Charge de travail prévue", value=850)
        acwr = float(acuteLoad / chronicLoad)
        st.metric("Ratio ACWR", f"{acwr:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><div class="card-title">🤖 Résultat IA</div>', unsafe_allow_html=True)
        if st.button("🚀 Lancer le Diagnostic Global", type="primary", use_container_width=True):
            payload = {
                "playerId": 10, "totalLoad": totalLoad, "sommeil": sommeil, "fatigue": fatigue,
                "douleurMusculaire": douleur, "stress": stress, "acuteLoad": acuteLoad,
                "chronicLoad": chronicLoad, "ACWR": acwr, "model": model_choice
            }
            with st.spinner("Analyse en cours..."):
                try:
                    res = requests.post(API_URL_GLOBAL, json=payload)
                    if res.status_code == 200:
                        data = res.json()
                        risk = data['riskScore']
                        st.markdown(f"<h1 style='font-size: 3rem; color: #1e3c72; text-align: center;'>{risk*100:.0f}%</h1>", unsafe_allow_html=True)
                        st.progress(risk)
                        
                        df_factors = pd.DataFrame(data['factors']).sort_values(by="contribution", ascending=True)
                        st.bar_chart(df_factors.set_index("feature"), horizontal=True, use_container_width=True)
                    else:
                        st.error("Erreur API")
                except requests.exceptions.ConnectionError:
                    st.error("API hors ligne. Lance uvicorn main:app")
        st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# PAGE 2 : CARTOGRAPHIE DES ZONES (Modèle 2)
# ==========================================
elif page == "🗺️ Cartographie Anatomique (Zones)":
    st.markdown("""
    <div class="header-container">
        <div class="logo">🗺️</div>
        <div>
            <h1>ERP Club AI - Cartographie Anatomique</h1>
            <span style="opacity: 0.9;">Classification Multi-classe (Prédiction de la localisation anatomique)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 📋 Profil Physique")
        position = st.selectbox("Position", ["Attaquant", "Milieu", "Défenseur", "Gardien"])
        foot = st.selectbox("Pied Fort", ["Droitier", "Gaucher", "Ambidextre"])
        age = st.slider("Âge", 16, 40, 24)
        fifa = st.slider("Note Générale", 50, 95, 75)

    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.markdown('<div class="card"><div class="card-title">🏃‍♂️ Data GPS & Médicales</div>', unsafe_allow_html=True)
        acute_zone = st.number_input("Acute Load (7j)", value=6000, step=100)
        chronic_zone = st.number_input("Chronic Load (28j)", value=4500, step=100)
        acwr_zone = float(acute_zone / chronic_zone) if chronic_zone > 0 else 0
        st.metric("ACWR", f"{acwr_zone:.2f}")
        
        douleur_z = st.slider("Douleurs Musculaires (1-10)", 1.0, 10.0, 4.0)
        souplesse_z = st.slider("Souplesse (1-10)", 1.0, 10.0, 6.0)
        agilite_z = st.slider("Agilité (1-10)", 1.0, 10.0, 8.0)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><div class="card-title">🧬 Analyse de Zone</div>', unsafe_allow_html=True)
        
        if st.button("🔥 Cartographier le Risque de Blessure", type="primary", use_container_width=True):
            payload_zone = {
                "playerId": 10, "position": position, "foot": foot, "age": age, "fifa_rating": fifa,
                "acuteLoad": acute_zone, "chronicLoad": chronic_zone, "ACWR": acwr_zone,
                "douleurMusculaire": douleur_z, "souplesse": souplesse_z, "agilite": agilite_z
            }
            
            with st.spinner("🧠 Interrogation du modèle via FastAPI..."):
                try:
                    res_zone = requests.post(API_URL_ZONE, json=payload_zone)
                    if res_zone.status_code == 200:
                        predictions = res_zone.json()["predictions"]
                        
                        df_res = pd.DataFrame(list(predictions.items()), columns=["Zone", "Proba"])
                        df_res["Probabilité"] = df_res["Proba"] * 100
                        df_res = df_res.sort_values(by='Probabilité', ascending=False)
                        
                        top_zone = df_res.iloc[0]
                        st.markdown(f"""
                        <div class="danger-alert">
                            <div style="font-weight: 800; font-size: 1.2rem;">🚨 ZONE CRITIQUE : {top_zone['Zone']} ({top_zone['Probabilité']:.1f}%)</div>
                            En cas de lésion, le modèle estime que cette zone est statistiquement la plus exposée compte tenu du profil athlétique.
                        </div>
                        """, unsafe_allow_html=True)
                        
                       # ----- DÉBUT DE LA HEATMAP ANATOMIQUE -----
                        st.markdown("### 🧍 Heatmap Corporelle")
                        
                        # Fonction pour générer la couleur (du blanc au rouge vif selon la proba)
                        def get_heat_color(zone_name):
                            proba = predictions.get(zone_name, 0)
                            # On multiplie par 2.5 pour rendre les couleurs plus visibles
                            alpha = min(1.0, proba * 2.5) 
                            return f"rgba(239, 68, 68, {alpha})"
                            
                        # HTML SANS INDENTATION (très important pour que Streamlit ne le lise pas comme du texte)
                        heatmap_html = f"""
<div style="display: flex; flex-direction: column; align-items: center; gap: 4px; margin-bottom: 20px;">
    <div class="body-part" style="width: 50px; height: 50px; border-radius: 50%; background: {get_heat_color('TETE')};">TÊTE</div>
    
    <div style="display: flex; gap: 4px; align-items: flex-start;">
        <div class="body-part" style="width: 40px; height: 30px; border-radius: 10px 0 0 10px; background: {get_heat_color('EPAULE')};">ÉPAULE</div>
        <div class="body-part" style="width: 80px; height: 90px; border-radius: 8px; background: {get_heat_color('DOS')};">DOS</div>
        <div class="body-part" style="width: 40px; height: 30px; border-radius: 0 10px 10px 0; background: {get_heat_color('EPAULE')};">ÉPAULE</div>
    </div>
    
    <div style="display: flex; gap: 4px; margin-top: -55px;">
        <div class="body-part" style="width: 30px; height: 80px; border-radius: 15px; margin-right: 48px; background: {get_heat_color('BRAS')}; writing-mode: vertical-rl;">BRAS</div>
        <div class="body-part" style="width: 70px; height: 40px; border-radius: 8px; margin-top: 55px; background: {get_heat_color('HANCHE')};">HANCHE</div>
        <div class="body-part" style="width: 30px; height: 80px; border-radius: 15px; margin-left: 48px; background: {get_heat_color('BRAS')}; writing-mode: vertical-rl;">BRAS</div>
    </div>
    
    <div style="display: flex; gap: 4px; align-items: flex-start;">
        <div class="body-part" style="width: 25px; height: 30px; border-radius: 50%; margin-right: 12px; background: {get_heat_color('MAIN')};">MAIN</div>
        <div class="body-part" style="width: 33px; height: 70px; border-radius: 8px; background: {get_heat_color('CUISSE')};">CUISSE</div>
        <div class="body-part" style="width: 33px; height: 70px; border-radius: 8px; background: {get_heat_color('CUISSE')};">CUISSE</div>
        <div class="body-part" style="width: 25px; height: 30px; border-radius: 50%; margin-left: 12px; background: {get_heat_color('MAIN')};">MAIN</div>
    </div>
    
    <div style="display: flex; gap: 4px;">
        <div class="body-part" style="width: 30px; height: 30px; border-radius: 50%; background: {get_heat_color('GENOU')};">GENOU</div>
        <div class="body-part" style="width: 30px; height: 30px; border-radius: 50%; background: {get_heat_color('GENOU')};">GENOU</div>
    </div>
    
    <div style="display: flex; gap: 4px;">
        <div class="body-part" style="width: 28px; height: 60px; border-radius: 8px; background: {get_heat_color('JAMBE')};">JAMBE</div>
        <div class="body-part" style="width: 28px; height: 60px; border-radius: 8px; background: {get_heat_color('JAMBE')};">JAMBE</div>
    </div>
    
    <div style="display: flex; gap: 4px;">
        <div class="body-part" style="width: 25px; height: 20px; border-radius: 8px; background: {get_heat_color('CHEVILLE')};">CHEV.</div>
        <div class="body-part" style="width: 25px; height: 20px; border-radius: 8px; background: {get_heat_color('CHEVILLE')};">CHEV.</div>
    </div>
    
    <div style="display: flex; gap: 4px;">
        <div class="body-part" style="width: 35px; height: 20px; border-radius: 10px 10px 0 0; background: {get_heat_color('PIED')};">PIED</div>
        <div class="body-part" style="width: 35px; height: 20px; border-radius: 10px 10px 0 0; background: {get_heat_color('PIED')};">PIED</div>
    </div>
</div>
"""
                        st.markdown(heatmap_html.replace('\n', ''), unsafe_allow_html=True)
                        # ----- FIN DE LA HEATMAP ANATOMIQUE -----

                        # Graphique Radar
                        st.markdown("### 📊 Distribution des probabilités")
                        fig_radar = px.line_polar(df_res.head(8), r='Probabilité', theta='Zone', line_close=True, color_discrete_sequence=['#ef4444'])
                        fig_radar.update_traces(fill='toself', fillcolor='rgba(239, 68, 68, 0.4)')
                        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), margin=dict(l=30, r=30, t=20, b=20))
                        st.plotly_chart(fig_radar, use_container_width=True)
                        
                    else:
                        st.error(f"Erreur Serveur: {res_zone.text}")
                except requests.exceptions.ConnectionError:
                    st.error("API FastAPI hors ligne. Vérifie que Uvicorn tourne sur le port 8000 !")
        else:
            st.info("Configurez le joueur et lancez l'IA pour voir la Heatmap anatomique simulée.")
        
        st.markdown('</div>', unsafe_allow_html=True)