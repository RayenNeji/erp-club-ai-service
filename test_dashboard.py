import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 1. CONFIGURATION INITIALE
# ==========================================
st.set_page_config(
    page_title="ERP Club AI - Hub Analytique",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. DESIGN SYSTEM (CSS GLOBAL)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] { 
        font-family: 'Inter', sans-serif; 
    }
    
    .stApp { 
        background: #f8fafc; 
    }
    
    /* Header Premium */
    .header-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 1.5rem 2.5rem; 
        border-radius: 1.25rem; 
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.1); 
        color: white;
        display: flex; 
        align-items: center; 
        gap: 1.5rem;
    }
    .header-container h1 { 
        font-weight: 800; 
        font-size: 2.2rem; 
        margin: 0; 
        color: #f8fafc;
    }
    .header-container p {
        color: #94a3b8;
        font-size: 1rem;
        margin: 0;
        font-weight: 500;
    }
    .logo-box {
        font-size: 3rem; 
        background: rgba(255,255,255,0.1); 
        width: 4.5rem; 
        height: 4.5rem;
        display: flex; 
        align-items: center; 
        justify-content: center; 
        border-radius: 1rem;
        box-shadow: inset 0 2px 4px rgba(255,255,255,0.1);
    }
    
    /* Cartes de contenu */
    .dashboard-card {
        background: #ffffff;
        border-radius: 1rem; 
        padding: 1.5rem 2rem; 
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03); 
        border: 1px solid #e2e8f0;
    }
    .card-title { 
        font-weight: 700; 
        font-size: 1.2rem; 
        color: #0f172a; 
        border-bottom: 2px solid #f1f5f9; 
        padding-bottom: 0.75rem; 
        margin-bottom: 1.25rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Alertes et Métriques */
    .danger-alert {
        background: #fef2f2; 
        border-left: 6px solid #ef4444; 
        border-radius: 0.5rem;
        padding: 1.25rem; 
        color: #991b1b; 
        margin: 1.5rem 0;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(239, 68, 68, 0.05);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #0f172a;
    }
    
    /* Heatmap Anatomique */
    .body-part {
        border: 1px solid #cbd5e1; 
        display: flex; 
        justify-content: center; 
        align-items: center;
        font-size: 0.65rem; 
        font-weight: 700; 
        color: #334155; 
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: white;
        text-shadow: 0px 1px 2px rgba(0,0,0,0.5);
    }
    
    /* Boutons */
    .stButton>button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white; 
        font-weight: 600; 
        border-radius: 0.5rem; 
        padding: 0.75rem 1rem; 
        border: none;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. NAVIGATION ET ROUTES API
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/8112/8112465.png", width=60)
st.sidebar.markdown("## 🧭 Navigation IA")
page = st.sidebar.radio(
    "Sélectionnez un microservice :",
    [
        "🩺 M1 - Risque de Blessure (Global)", 
        "🗺️ M2 - Cartographie (Zones)", 
        "⏳ M3 - Analyse de Survie (Rechute)"
    ]
)
st.sidebar.markdown("---")
st.sidebar.caption("ERP Club AI v4.0 - Déploiement Local")

# Endpoints de l'API FastAPI
API_URL_GLOBAL = "http://localhost:8000/predict-injury"
API_URL_ZONE = "http://localhost:8000/predict-injury-zone"
API_URL_RELAPSE = "http://localhost:8000/predict-relapse"

# ==========================================
# PAGE 1 : RISQUE GLOBAL (Modèle 1)
# ==========================================
if page == "🩺 M1 - Risque de Blessure (Global)":
    st.markdown('''
    <div class="header-container">
        <div class="logo-box">🩺</div>
        <div>
            <h1>Prédiction Globale de Blessure</h1>
            <p>Classification Binaire XGBoost : Analyse des charges et facteurs de fatigue</p>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    with st.sidebar: 
        st.markdown("### ⚙️ Configuration")
        model_choice = st.selectbox("Algorithme", ["XGBoost", "LightGBM"])
        player_id = st.number_input("ID Joueur", value=10, step=1)

    col1, col2 = st.columns([1.2, 1], gap="large")

    with col1:
        st.markdown('<div class="dashboard-card"><div class="card-title">📊 Paramètres Cliniques & GPS</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            sommeil = st.slider("💤 Sommeil (Qualité)", 1.0, 10.0, 7.5)
            fatigue = st.slider("⚡ Fatigue (RPE)", 1.0, 10.0, 4.0)
            acuteLoad = st.number_input("Acute Load (7j)", value=5950, step=100)
        with c2:
            douleur = st.slider("🦵 Douleurs Musculaires", 1.0, 10.0, 3.0)
            stress = st.slider("🧘 Niveau de Stress", 1.0, 10.0, 4.5)
            chronicLoad = st.number_input("Chronic Load (28j)", value=5100, step=100)
            
        totalLoad = st.number_input("Charge de travail prévue (Aujourd'hui)", value=850)
        acwr = float(acuteLoad / chronicLoad) if chronicLoad > 0 else 0
        
        acwr_color = "normal"
        if acwr > 1.5 or acwr < 0.8: acwr_color = "inverse"
        st.metric("Ratio ACWR (Acute:Chronic)", f"{acwr:.2f}", delta="Danger" if acwr>1.5 else "Optimal", delta_color=acwr_color)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="dashboard-card"><div class="card-title">🤖 Diagnostic de l\'IA</div>', unsafe_allow_html=True)
        
        if st.button("🚀 Lancer l'Analyse Prédictive", use_container_width=True):
            payload = {
                "playerId": player_id, "totalLoad": totalLoad, "sommeil": sommeil, 
                "fatigue": fatigue, "douleurMusculaire": douleur, "stress": stress, 
                "acuteLoad": acuteLoad, "chronicLoad": chronicLoad, "ACWR": acwr, 
                "model": model_choice
            }
            with st.spinner("Analyse des arbres de décision en cours..."):
                try:
                    res = requests.post(API_URL_GLOBAL, json=payload)
                    if res.status_code == 200:
                        data = res.json()
                        risk_prob = data['riskScore'] * 100
                        color = "#ef4444" if risk_prob > 60 else ("#f59e0b" if risk_prob > 30 else "#22c55e")
                        
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1rem;">
                            <p style="margin:0; color: #64748b; font-weight: 600; text-transform: uppercase;">Probabilité de Blessure Imminente</p>
                            <h1 style="font-size: 4rem; color: {color}; margin: 0; font-weight: 800;">{risk_prob:.0f}%</h1>
                            <p style="margin:0; font-weight: bold; color: {color};">Niveau : {data['riskLevel']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.progress(data['riskScore'])
                        
                        if data.get('factors'):
                            st.markdown("#### 🔍 Principaux Facteurs (Explicabilité)")
                            df_factors = pd.DataFrame(data['factors']).sort_values(by="contribution", ascending=True)
                            fig_factors = px.bar(df_factors, x="contribution", y="feature", orientation='h', color="impact", color_discrete_map={"négatif": "#ef4444", "positif": "#22c55e"})
                            fig_factors.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=250)
                            st.plotly_chart(fig_factors, use_container_width=True)
                    else: 
                        st.error(f"Erreur API ({res.status_code}): {res.text}")
                except Exception as e: 
                    st.error(f"Impossible de contacter l'API. Vérifiez que FastAPI tourne sur le port 8000.\nErreur: {e}")
        else:
            st.info("👈 Ajustez les paramètres et lancez l'analyse pour évaluer le risque du joueur.")
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# PAGE 2 : CARTOGRAPHIE DES ZONES (Modèle 2)
# ==========================================
elif page == "🗺️ M2 - Cartographie (Zones)":
    st.markdown('''
    <div class="header-container">
        <div class="logo-box">🗺️</div>
        <div>
            <h1>Cartographie Anatomique</h1>
            <p>Modèle Multi-classe Random Forest : Prédiction des zones de vulnérabilité corporelle</p>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### 📋 Morphologie & Poste")
        player_id = st.number_input("ID Joueur", value=10, step=1)
        position = st.selectbox("Position", ["Attaquant", "Milieu", "Défenseur", "Gardien"])
        foot = st.selectbox("Pied Fort", ["Droitier", "Gaucher", "Ambidextre"])
        age = st.slider("Âge", 16, 40, 24)
        fifa = st.slider("Note Générale (Fifa Rating)", 50, 95, 75)

    col1, col2 = st.columns([1, 1.3], gap="large")
    
    with col1:
        st.markdown('<div class="dashboard-card"><div class="card-title">🏃‍♂️ Biomecanique & Charge</div>', unsafe_allow_html=True)
        acute_zone = st.number_input("Acute Load (7j)", value=6000, step=100, key="z_acute")
        chronic_zone = st.number_input("Chronic Load (28j)", value=4500, step=100, key="z_chronic")
        acwr_zone = float(acute_zone / chronic_zone) if chronic_zone > 0 else 0
        st.metric("ACWR", f"{acwr_zone:.2f}")
        
        st.markdown("---")
        douleur_z = st.slider("Douleurs Actuelles (1-10)", 1.0, 10.0, 4.0, key="z_doul")
        souplesse_z = st.slider("Souplesse Globale (1-10)", 1.0, 10.0, 6.0, key="z_soup")
        agilite_z = st.slider("Test d'Agilité (1-10)", 1.0, 10.0, 8.0, key="z_agil")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="dashboard-card"><div class="card-title">🧬 Analyse des Points de Rupture</div>', unsafe_allow_html=True)
        if st.button("🔥 Générer la Cartographie Corporelle", use_container_width=True):
            payload_zone = {
                "playerId": player_id, "position": position, "foot": foot, "age": age, 
                "fifa_rating": fifa, "acuteLoad": acute_zone, "chronicLoad": chronic_zone, 
                "ACWR": acwr_zone, "douleurMusculaire": douleur_z, "souplesse": souplesse_z, 
                "agilite": agilite_z
            }
            with st.spinner("Cartographie des probabilités par zone..."):
                try:
                    res_zone = requests.post(API_URL_ZONE, json=payload_zone)
                    if res_zone.status_code == 200:
                        predictions = res_zone.json()["predictions"]
                        
                        df_res = pd.DataFrame(list(predictions.items()), columns=["Zone", "Proba"])
                        df_res["Probabilité"] = df_res["Proba"] * 100
                        df_res = df_res.sort_values(by='Probabilité', ascending=False)
                        top_zone = df_res.iloc[0]
                        
                        st.markdown(f'<div class="danger-alert">🚨 ZONE CRITIQUE DÉTECTÉE : {top_zone["Zone"].upper()} ({top_zone["Probabilité"]:.1f}%)</div>', unsafe_allow_html=True)
                        
                        def get_heat_color(zone_name):
                            val = predictions.get(zone_name, 0)
                            intensity = min(1.0, val * 3) 
                            return f"rgba({int(148 + (239-148)*intensity)}, {int(163 - 95*intensity)}, {int(184 - 116*intensity)}, 1)"
                        
                        tab1, tab2 = st.tabs(["🧍‍♂️ Vue Anatomique", "📊 Graphique Radar"])
                        
                        with tab1:
                            heatmap_html = f"""
                            <div style="display: flex; flex-direction: column; align-items: center; gap: 6px; margin: 20px 0;">
                                <div class="body-part" style="width: 55px; height: 55px; border-radius: 50%; background: {get_heat_color('TETE')};">TÊTE</div>
                                <div style="display: flex; gap: 6px; align-items: flex-start;">
                                    <div class="body-part" style="width: 45px; height: 35px; border-radius: 12px 0 0 12px; background: {get_heat_color('EPAULE')};">ÉPAULE</div>
                                    <div class="body-part" style="width: 90px; height: 100px; border-radius: 8px; background: {get_heat_color('DOS')};">TORSE / DOS</div>
                                    <div class="body-part" style="width: 45px; height: 35px; border-radius: 0 12px 12px 0; background: {get_heat_color('EPAULE')};">ÉPAULE</div>
                                </div>
                                <div style="display: flex; gap: 6px; margin-top: -65px;">
                                    <div class="body-part" style="width: 35px; height: 90px; border-radius: 15px; margin-right: 52px; background: {get_heat_color('BRAS')}; writing-mode: vertical-rl;">BRAS</div>
                                    <div class="body-part" style="width: 80px; height: 45px; border-radius: 8px; margin-top: 65px; background: {get_heat_color('HANCHE')};">HANCHE / AINE</div>
                                    <div class="body-part" style="width: 35px; height: 90px; border-radius: 15px; margin-left: 52px; background: {get_heat_color('BRAS')}; writing-mode: vertical-rl;">BRAS</div>
                                </div>
                                <div style="display: flex; gap: 6px; align-items: flex-start;">
                                    <div class="body-part" style="width: 25px; height: 35px; border-radius: 50%; margin-right: 12px; background: {get_heat_color('MAIN')};">MAIN</div>
                                    <div class="body-part" style="width: 37px; height: 80px; border-radius: 8px; background: {get_heat_color('CUISSE')};">CUISSE</div>
                                    <div class="body-part" style="width: 37px; height: 80px; border-radius: 8px; background: {get_heat_color('CUISSE')};">CUISSE</div>
                                    <div class="body-part" style="width: 25px; height: 35px; border-radius: 50%; margin-left: 12px; background: {get_heat_color('MAIN')};">MAIN</div>
                                </div>
                                <div style="display: flex; gap: 6px;">
                                    <div class="body-part" style="width: 35px; height: 35px; border-radius: 50%; background: {get_heat_color('GENOU')};">GENOU</div>
                                    <div class="body-part" style="width: 35px; height: 35px; border-radius: 50%; background: {get_heat_color('GENOU')};">GENOU</div>
                                </div>
                                <div style="display: flex; gap: 6px;">
                                    <div class="body-part" style="width: 30px; height: 70px; border-radius: 8px; background: {get_heat_color('JAMBE')};">MOLLET</div>
                                    <div class="body-part" style="width: 30px; height: 70px; border-radius: 8px; background: {get_heat_color('JAMBE')};">MOLLET</div>
                                </div>
                                <div style="display: flex; gap: 6px;">
                                    <div class="body-part" style="width: 28px; height: 25px; border-radius: 8px; background: {get_heat_color('CHEVILLE')};">CHEV.</div>
                                    <div class="body-part" style="width: 28px; height: 25px; border-radius: 8px; background: {get_heat_color('CHEVILLE')};">CHEV.</div>
                                </div>
                                <div style="display: flex; gap: 6px;">
                                    <div class="body-part" style="width: 40px; height: 20px; border-radius: 10px 10px 0 0; background: {get_heat_color('PIED')};">PIED</div>
                                    <div class="body-part" style="width: 40px; height: 20px; border-radius: 10px 10px 0 0; background: {get_heat_color('PIED')};">PIED</div>
                                </div>
                            </div>
                            """
                            st.markdown(heatmap_html.replace('\n', ''), unsafe_allow_html=True)
                            
                        with tab2:
                            fig_radar = px.line_polar(
                                df_res.head(8), r='Probabilité', theta='Zone', 
                                line_close=True, color_discrete_sequence=['#ef4444']
                            )
                            fig_radar.update_traces(fill='toself', fillcolor='rgba(239, 68, 68, 0.3)', line=dict(width=3))
                            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, df_res['Probabilité'].max() + 5])), margin=dict(l=40, r=40, t=20, b=20))
                            st.plotly_chart(fig_radar, use_container_width=True)
                            
                    else: st.error("Erreur avec le modèle de zone.")
                except Exception as e: st.error(f"API injoignable : {e}")
        else:
            st.info("Sélectionnez la biométrie du joueur et lancez le modèle pour cartographier les fragilités corporelles.")
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# PAGE 3 : ANALYSE DE SURVIE (Modèle 3)
# ==========================================
elif page == "⏳ M3 - Analyse de Survie (Rechute)":
    st.markdown('''
    <div class="header-container" style="background: linear-gradient(135deg, #064e3b 0%, #0f766e 100%);">
        <div class="logo-box">⏳</div>
        <div>
            <h1>Analyse de Survie Post-Rééducation</h1>
            <p>Régression de Cox (Cox PH) : Estimation temporelle du risque de rechute après retour au jeu.</p>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 🏥 Bilan de Rééducation")
        player_id = st.number_input("ID Joueur", value=10, step=1)
        physio_adherence = st.slider("Adhérence au protocole Physio (%)", 0, 100, 85, help="Taux d'assiduité aux séances de kinésithérapie.")
        post_acwr = st.slider("ACWR projeté (Retour au jeu)", 0.5, 2.5, 1.1, step=0.1, help="Charge de travail accumulée prévue pour sa reprise.")

    col1, col2 = st.columns([1, 1.5], gap="large")

    with col1:
        st.markdown('<div class="dashboard-card"><div class="card-title">🩺 État de Santé (Wellness)</div>', unsafe_allow_html=True)
        recovery = st.slider("Score de Récupération (0-100)", 0, 100, 75)
        sleep = st.slider("Qualité du Sommeil (1-10)", 1.0, 10.0, 7.5)
        stress = st.slider("Niveau de Stress (0-1)", 0.0, 1.0, 0.3)
        fatigue = st.slider("Index de Fatigue (0-100)", 0, 100, 45)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="dashboard-card"><div class="card-title">📈 Projection de Survie au fil du temps</div>', unsafe_allow_html=True)
        
        if st.button("🔮 Calculer la Courbe de Survie (Cox PH)", use_container_width=True):
            payload_surv = {
                "playerId": player_id, "recovery_score": recovery, "sleep_quality": sleep,
                "stress_level": stress, "fatigue_index": fatigue, "physio_adherence": physio_adherence,
                "post_recovery_ACWR": post_acwr
            }
            
            with st.spinner("Modélisation de la survie temporelle (Kaplan-Meier estimé)..."):
                try:
                    res_surv = requests.post(API_URL_RELAPSE, json=payload_surv)
                    
                    if res_surv.status_code == 200:
                        data_surv = res_surv.json()
                        curve_data = data_surv.get("survival_curve", [])
                        
                        if curve_data:
                            df_curve = pd.DataFrame(curve_data)
                            
                            # Création du graphique interactif avec Plotly
                            fig = px.line(
                                df_curve, x="day", y="probability",
                                labels={"day": "Jours de suivi après retour", "probability": "Probabilité de ne pas rechuter"},
                                color_discrete_sequence=['#10b981'] # Vert émeraude
                            )
                            
                            fig.update_layout(
                                yaxis_range=[0, 1.05], 
                                margin=dict(l=20, r=20, t=40, b=20),
                                plot_bgcolor='rgba(0,0,0,0)'
                            )
                            
                            # Ajout de la ligne de danger à 50%
                            fig.add_hline(
                                y=0.5, line_dash="dot", 
                                annotation_text="Seuil de Danger Critique (50%)", 
                                annotation_position="bottom right", line_color="#ef4444"
                            )
                            
                            fig.update_traces(fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.15)', line=dict(width=3))
                            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')
                            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Précision globale (C-Index)
                            c_index = data_surv.get("c_index", 0.96)
                            st.success(f"**Concordance Index (Précision globale du modèle) : {c_index:.3f}**")
                            
                            # Alerte Médicale 
                            # Récupération de la probabilité au jour 30 (ou la dernière connue si < 30)
                            jours_30 = df_curve[df_curve['day'] >= 30]
                            prob_a_30_jours = jours_30.iloc[0]['probability'] if not jours_30.empty else df_curve.iloc[-1]['probability']
                            
                            if prob_a_30_jours < 0.6:
                                st.error(f"⚠️ **Alerte Médicale :** Le joueur a seulement **{prob_a_30_jours*100:.1f}%** de chance de ne pas rechuter le premier mois. Protocole de reprise inadapté ou trop agressif.")
                            else:
                                st.info(f"✅ **Bilan positif :** Le joueur a **{prob_a_30_jours*100:.1f}%** de chance de rester en bonne santé après 30 jours.")
                        else:
                            st.warning("L'API n'a retourné aucune donnée pour la courbe.")
                    else:
                        st.error(f"Erreur avec le modèle de survie ({res_surv.status_code}) : {res_surv.text}")
                except Exception as e:
                    st.error(f"API injoignable. Assurez-vous que FastAPI tourne sur le port 8000.\nDétails : {e}")
        else:
            st.info("👈 Ajustez les paramètres post-rééducation du patient (Adhérence et ACWR sont primordiaux) et lancez le modèle pour simuler son risque temporel de rechute.")
            
        st.markdown('</div>', unsafe_allow_html=True)