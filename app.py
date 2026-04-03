import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Configuration page ────────────────────────────────────────
st.set_page_config(
    page_title="Smart City — Prévision PM2.5 Beijing",
    page_icon="🏙️",
    layout="wide"
)

# ── CSS personnalisé ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ═══ SIDEBAR ═══════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080c10 0%, #0b1018 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] > div:first-child { padding: 0; }

.sidebar-brand {
    display: flex; align-items: center; gap: 12px;
    padding: 28px 20px 22px 20px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 10px;
}
.sidebar-brand-icon {
    width: 38px; height: 38px; border-radius: 10px;
    background: linear-gradient(135deg, #1D9E75, #0d6e51);
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
    box-shadow: 0 0 20px rgba(29,158,117,0.3);
}
.sidebar-brand-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 15px; font-weight: 600; color: #f0f4f8;
    line-height: 1.2; letter-spacing: -0.3px;
}
.sidebar-brand-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: #4b5a6a; font-weight: 400; margin-top: 2px;
    letter-spacing: 0.05em;
}

.sidebar-badge {
    display: inline-flex; align-items: center; gap: 7px;
    background: rgba(29,158,117,0.08); color: #34d399;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; font-weight: 500;
    padding: 5px 12px; border-radius: 20px;
    margin: 0 20px 18px 20px;
    border: 1px solid rgba(29,158,117,0.2);
}
.sidebar-badge-dot {
    width: 6px; height: 6px; border-radius: 50%; background: #1D9E75;
    box-shadow: 0 0 6px #1D9E75;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; } 50% { opacity: 0.4; }
}

.sidebar-section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; font-weight: 500; text-transform: uppercase;
    letter-spacing: 0.12em; color: #2d3748; padding: 0 20px 10px 20px;
}

[data-testid="stSidebar"] .stRadio > label { display: none; }
[data-testid="stSidebar"] .stRadio > div { gap: 1px !important; display: flex; flex-direction: column; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    display: flex !important; align-items: center !important;
    padding: 10px 20px !important; border-radius: 0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important; font-weight: 400 !important;
    color: #4b5a6a !important; cursor: pointer !important;
    margin: 0 !important; border: none !important; background: transparent !important;
    transition: all 0.15s !important;
    border-left: 2px solid transparent !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.03) !important; color: #94a3b8 !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] input[type="radio"] { display: none !important; }

[data-testid="stSidebar"] img {
    border-radius: 12px; margin: 0 20px;
    width: calc(100% - 40px) !important;
    object-fit: cover; height: 120px; margin-bottom: 18px;
    opacity: 0.85;
    border: 1px solid rgba(255,255,255,0.07);
}

.sidebar-metrics {
    margin: 0 16px 20px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 10px; padding: 14px;
}
.sidebar-metrics-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: #2d3748; text-transform: uppercase;
    letter-spacing: 0.1em; margin-bottom: 10px; font-weight: 500;
}
.sidebar-metric-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 5px 0; border-bottom: 1px solid rgba(255,255,255,0.03);
}
.sidebar-metric-label { font-family: 'DM Sans', sans-serif; font-size: 11px; color: #4b5a6a; }
.sidebar-metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; font-weight: 500; color: #94a3b8;
}

/* ═══ PAGE PRINCIPALE ═══════════════════════════════════════ */
.main .block-container {
    padding-top: 2.5rem; padding-left: 3rem; padding-right: 3rem;
    max-width: 1400px;
}

[data-testid="stMetric"] {
    background: #ffffff; border: 1px solid #e8ecf0;
    border-radius: 12px; padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
[data-testid="stMetricLabel"] { font-size: 11px !important; color: #94a3b8 !important; font-weight: 500 !important; text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="stMetricValue"] { font-size: 22px !important; font-weight: 700 !important; color: #0f172a !important; letter-spacing: -0.5px; }

.stButton > button[kind="primary"] {
    background: #0f172a !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    font-size: 14px !important; padding: 12px 24px !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"]:hover {
    background: #1D9E75 !important; transform: translateY(-1px) !important;
}

/* ═══ RAPPORT A PROPOS ══════════════════════════════════════ */
.report-hero {
    background: linear-gradient(135deg, #080c10 0%, #0d1a14 50%, #080c10 100%);
    border-radius: 20px;
    padding: 56px 64px;
    margin-bottom: 40px;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(29,158,117,0.15);
}
.report-hero::before {
    content: '';
    position: absolute; top: -60px; right: -60px;
    width: 300px; height: 300px; border-radius: 50%;
    background: radial-gradient(circle, rgba(29,158,117,0.12) 0%, transparent 70%);
}
.report-hero-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: #1D9E75; letter-spacing: 0.15em;
    text-transform: uppercase; margin-bottom: 16px; font-weight: 500;
}
.report-hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 48px; color: #f0f4f8; line-height: 1.1;
    letter-spacing: -1px; margin-bottom: 12px; font-weight: 400;
}
.report-hero-title em {
    font-style: italic; color: #1D9E75;
}
.report-hero-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 16px; color: #4b5a6a; line-height: 1.6;
    max-width: 580px; font-weight: 400;
}
.report-hero-meta {
    display: flex; gap: 32px; margin-top: 32px; padding-top: 32px;
    border-top: 1px solid rgba(255,255,255,0.07);
}
.report-hero-meta-item { }
.report-hero-meta-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: #2d3748; text-transform: uppercase;
    letter-spacing: 0.1em; margin-bottom: 4px;
}
.report-hero-meta-value {
    font-family: 'DM Sans', sans-serif;
    font-size: 14px; color: #94a3b8; font-weight: 400;
}

/* Section headers */
.report-section-header {
    display: flex; align-items: center; gap: 16px;
    margin: 48px 0 28px 0;
}
.report-section-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: #1D9E75; font-weight: 500;
    background: rgba(29,158,117,0.08);
    border: 1px solid rgba(29,158,117,0.2);
    padding: 4px 10px; border-radius: 20px;
    letter-spacing: 0.05em;
}
.report-section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 28px; color: #0f172a; font-weight: 400;
    letter-spacing: -0.5px;
}
.report-section-line {
    flex: 1; height: 1px; background: #e8ecf0;
}

/* Stat cards grand format */
.stat-cards-grid {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;
    margin-bottom: 32px;
}
.stat-card {
    background: #ffffff;
    border: 1px solid #e8ecf0;
    border-radius: 16px; padding: 28px 24px;
    position: relative; overflow: hidden;
}
.stat-card-accent {
    position: absolute; top: 0; left: 0; right: 0;
    height: 3px;
}
.stat-card-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: #94a3b8; text-transform: uppercase;
    letter-spacing: 0.1em; margin-bottom: 12px;
}
.stat-card-value {
    font-family: 'DM Serif Display', serif;
    font-size: 42px; color: #0f172a; line-height: 1;
    letter-spacing: -1px; margin-bottom: 6px;
}
.stat-card-unit {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px; color: #94a3b8;
}
.stat-card-desc {
    font-family: 'DM Sans', sans-serif;
    font-size: 12px; color: #64748b; margin-top: 12px;
    line-height: 1.5; padding-top: 12px;
    border-top: 1px solid #f1f5f9;
}

/* Timeline méthodologique */
.timeline { position: relative; padding-left: 32px; }
.timeline::before {
    content: ''; position: absolute; left: 8px; top: 8px; bottom: 8px;
    width: 1px; background: linear-gradient(to bottom, #1D9E75, #e8ecf0);
}
.timeline-item { position: relative; margin-bottom: 28px; }
.timeline-dot {
    position: absolute; left: -28px; top: 6px;
    width: 16px; height: 16px; border-radius: 50%;
    background: #1D9E75; border: 3px solid #f8fafc;
    box-shadow: 0 0 0 1px #1D9E75;
}
.timeline-step {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: #1D9E75; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px;
}
.timeline-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 15px; font-weight: 600; color: #0f172a; margin-bottom: 6px;
}
.timeline-desc {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px; color: #64748b; line-height: 1.6;
}

/* Feature chips */
.feature-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
.feature-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: #334155;
    background: #f1f5f9; border: 1px solid #e2e8f0;
    padding: 4px 10px; border-radius: 6px;
}

/* Tableau comparatif stylisé */
.model-table { width: 100%; border-collapse: collapse; margin-top: 16px; }
.model-table th {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em;
    color: #94a3b8; padding: 12px 16px; text-align: left;
    border-bottom: 1px solid #e8ecf0; font-weight: 500;
}
.model-table td {
    font-family: 'DM Sans', sans-serif;
    font-size: 14px; color: #334155;
    padding: 14px 16px; border-bottom: 1px solid #f1f5f9;
}
.model-table tr.winner { background: #f0fdf8; }
.model-table tr.winner td { color: #0f172a; font-weight: 600; }
.winner-badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: #1D9E75;
    background: rgba(29,158,117,0.1);
    border: 1px solid rgba(29,158,117,0.3);
    padding: 2px 8px; border-radius: 20px;
    margin-left: 8px; vertical-align: middle;
}

/* Limite cards */
.limit-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
.limit-card {
    background: #fffbf0; border: 1px solid #fde68a;
    border-radius: 12px; padding: 18px 20px;
    border-left: 3px solid #f59e0b;
}
.limit-card-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px; font-weight: 600; color: #78350f; margin-bottom: 4px;
}
.limit-card-desc {
    font-family: 'DM Sans', sans-serif;
    font-size: 12px; color: #92400e; line-height: 1.5;
}

/* Footer auteur */
.author-card {
    display: flex; align-items: center; gap: 20px;
    background: #0f172a; border-radius: 16px;
    padding: 28px 32px; margin-top: 40px;
}
.author-avatar {
    width: 52px; height: 52px; border-radius: 50%;
    background: linear-gradient(135deg, #1D9E75, #185FA5);
    display: flex; align-items: center; justify-content: center;
    font-family: 'DM Sans', sans-serif;
    font-size: 18px; font-weight: 700; color: white; flex-shrink: 0;
}
.author-name {
    font-family: 'DM Serif Display', serif;
    font-size: 20px; color: #f0f4f8; margin-bottom: 2px;
}
.author-role {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: #4b5a6a; letter-spacing: 0.05em;
}
.author-link {
    font-family: 'DM Sans', sans-serif;
    font-size: 12px; color: #1D9E75; margin-top: 8px; display: block;
    text-decoration: none;
}
</style>
""", unsafe_allow_html=True)

# ── Chargement modèle & données ───────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("model_pm25.pkl")
    with open("features.json") as f:
        features = json.load(f)
    with open("model_stats.json") as f:
        stats = json.load(f)
    return model, features, stats

@st.cache_data
def load_data():
    df = pd.read_csv("beijing_features.csv",
                     index_col=0, parse_dates=True)
    return df

model, features, stats = load_model()
df = load_data()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.markdown("""
<div class="sidebar-brand">
    <div class="sidebar-brand-icon">
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
            <circle cx="10" cy="10" r="4" fill="white"/>
            <circle cx="10" cy="3" r="2" fill="white" opacity="0.5"/>
            <circle cx="10" cy="17" r="2" fill="white" opacity="0.5"/>
            <circle cx="3" cy="10" r="2" fill="white" opacity="0.5"/>
            <circle cx="17" cy="10" r="2" fill="white" opacity="0.5"/>
        </svg>
    </div>
    <div>
        <div class="sidebar-brand-title">Smart City</div>
        <div class="sidebar-brand-sub">Beijing PM2.5 — J+1</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="sidebar-badge">
    <div class="sidebar-badge-dot"></div>
    Modèle actif
</div>
""", unsafe_allow_html=True)

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/"
    "thumb/f/fa/Beijing_montage.jpg/320px-Beijing_montage.jpg",
    use_container_width=True
)

st.sidebar.markdown('<div class="sidebar-section-label">Navigation</div>',
                    unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    ["Accueil & Prediction",
     "Historique & Tendances",
     "Analyse des performances",
     "A propos du modele"]
)

st.sidebar.markdown(f"""
<div class="sidebar-metrics">
    <div class="sidebar-metric-row">
        <span class="sidebar-metric-label">RMSE</span>
        <span class="sidebar-metric-value">{stats['rmse']:.1f} µg/m³</span>
    </div>
    <div class="sidebar-metric-row">
        <span class="sidebar-metric-label">MAE</span>
        <span class="sidebar-metric-value">{stats['mae']:.1f} µg/m³</span>
    </div>
    <div class="sidebar-metric-row">
        <span class="sidebar-metric-label">R²</span>
        <span class="sidebar-metric-value">{stats['r2']:.3f}</span>
    </div>
    <div class="sidebar-metric-row" style="border-bottom:none;">
        <span class="sidebar-metric-label">Modèle</span>
        <span class="sidebar-metric-value">Random Forest</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 1 — Accueil & Prédiction
# ══════════════════════════════════════════════════════════════
if page == "Accueil & Prediction":

    st.title("🏙️ Prévision de la Pollution PM2.5 à Beijing")
    st.markdown(
        "Entrez les conditions météorologiques et de pollution "
        "d'aujourd'hui pour estimer le niveau de PM2.5 de demain."
    )

    # ── Métriques modèle en haut ──────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Modèle", "Random Forest")
    c2.metric("RMSE", f"{stats['rmse']:.1f} µg/m³")
    c3.metric("MAE",  f"{stats['mae']:.1f} µg/m³")
    c4.metric("R²",   f"{stats['r2']:.3f}")

    st.markdown("---")
    st.subheader("⚙️ Paramètres d'entrée")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🌫️ Pollution récente**")
        pm25_roll_3h  = st.slider(
            "PM2.5 moyen (3 dernières heures) µg/m³",
            0, 500, 80)
        pm25_lag_1h   = st.slider(
            "PM2.5 il y a 1h (µg/m³)", 0, 500, 75)
        pm25_lag_12h  = st.slider(                          # ✅ CORRIGÉ : slider dédié
            "PM2.5 il y a 12h (µg/m³)", 0, 500, 80)
        pm25_roll_12h = st.slider(
            "PM2.5 moyen (12 dernières heures) µg/m³",
            0, 500, 85)
        pm25_roll_24h = st.slider(
            "PM2.5 moyen (24 dernières heures) µg/m³",
            0, 500, 90)
        pm25_lag_6h   = st.slider(
            "PM2.5 il y a 6h (µg/m³)", 0, 500, 78)
        pm25_lag_24h  = st.slider(
            "PM2.5 il y a 24h (µg/m³)", 0, 500, 82)

    with col2:
        st.markdown("**🌡️ Météo**")
        TEMP  = st.slider("Température (°C)", -30, 45, 10)
        PRES  = st.slider("Pression (hPa)", 990, 1040, 1015)
        DEWP  = st.slider("Point de rosée (°C)", -40, 30, -5)
        Iws   = st.slider("Vitesse du vent (m/s)", 0, 200, 20)
        Ir    = st.slider("Heures de pluie cumulées", 0, 24, 0)
        Is    = st.slider("Heures de neige cumulées", 0, 24, 0)
        is_rainy = int(Ir > 0)

    with col3:
        st.markdown("**📅 Temporel & Vent**")
        month     = st.selectbox(
            "Mois", range(1, 13),
            format_func=lambda x: [
                "Jan","Fév","Mar","Avr","Mai","Jun",
                "Jul","Aoû","Sep","Oct","Nov","Déc"][x-1])
        dayofweek = st.selectbox(
            "Jour de la semaine",
            range(7),
            format_func=lambda x: [
                "Lundi","Mardi","Mercredi","Jeudi",
                "Ven","Sam","Dim"][x])
        is_weekend = int(dayofweek >= 5)
        wind_dir   = st.selectbox(
            "Direction du vent", ["NE","NW","SE","cv"])
        vent_speed = st.selectbox(
            "Catégorie vent", ["calme","modéré","fort"])

    # ── Calcul des features dérivées ──────────────────────────
    temp_feels_like = (
        TEMP + 0.33 * (DEWP / 100 * 6.105 *
        np.exp(17.27 * TEMP / (237.7 + TEMP))) - 4.0)
    delta_temp  = 0.0
    temp_roll_6h = float(TEMP)
    iws_roll_6h  = float(Iws)
    temp_x_vent  = TEMP * Iws
    pres_x_dewp  = PRES * DEWP

    wind_dir_NW = int(wind_dir == "NW")
    wind_dir_SE = int(wind_dir == "SE")
    wind_dir_cv = int(wind_dir == "cv")
    vent_modere = int(vent_speed == "modéré")
    vent_fort   = int(vent_speed == "fort")

    saison_map = {
        12:"Hiver", 1:"Hiver",  2:"Hiver",
        3:"Printemps", 4:"Printemps", 5:"Printemps",
        6:"Été",    7:"Été",    8:"Été",
        9:"Automne",10:"Automne",11:"Automne"
    }
    saison = saison_map[month]
    saison_Hiver     = int(saison == "Hiver")
    saison_Printemps = int(saison == "Printemps")
    saison_Ete       = int(saison == "Été")

    # Lags journaliers (non disponibles en temps réel → moyennes)
    pm25_lag_1d  = pm25_lag_24h
    pm25_lag_2d  = pm25_lag_24h
    pm25_lag_3d  = pm25_lag_24h
    pm25_lag_7d  = pm25_lag_24h
    pm25_roll_3d = pm25_roll_24h
    pm25_roll_7d = pm25_roll_24h
    pm25_roll_14d= pm25_roll_24h
    pm25_delta_1d= pm25_lag_1h - pm25_lag_24h

    # ── Construction du vecteur de features ───────────────────
    input_dict = {
        "pm25_lag_1h"     : pm25_lag_1h,
        "pm25_lag_6h"     : pm25_lag_6h,
        "pm25_lag_12h"    : pm25_lag_12h,   # ✅ CORRIGÉ : valeur indépendante
        "pm25_lag_24h"    : pm25_lag_24h,
        "pm25_roll_3h"    : pm25_roll_3h,
        "pm25_roll_12h"   : pm25_roll_12h,
        "pm25_roll_24h"   : pm25_roll_24h,
        "TEMP"            : TEMP,
        "PRES"            : PRES,
        "DEWP"            : DEWP,
        "Iws"             : Iws,
        "Is"              : Is,
        "Ir"              : Ir,
        "temp_feels_like" : temp_feels_like,
        "delta_temp"      : delta_temp,
        "temp_roll_6h"    : temp_roll_6h,
        "iws_roll_6h"     : iws_roll_6h,
        "temp_x_vent"     : temp_x_vent,
        "pres_x_dewp"     : pres_x_dewp,
        "is_rainy"        : is_rainy,
        "month"           : month,
        "dayofweek"       : dayofweek,
        "is_weekend"      : is_weekend,
        "wind_dir_NW"     : wind_dir_NW,
        "wind_dir_SE"     : wind_dir_SE,
        "wind_dir_cv"     : wind_dir_cv,
        "vent_modéré"     : vent_modere,
        "vent_fort"       : vent_fort,
        "saison_Hiver"    : saison_Hiver,
        "saison_Printemps": saison_Printemps,
        "saison_Été"      : saison_Ete,
        "pm25_lag_1d"     : pm25_lag_1d,
        "pm25_lag_2d"     : pm25_lag_2d,
        "pm25_lag_3d"     : pm25_lag_3d,
        "pm25_lag_7d"     : pm25_lag_7d,
        "pm25_roll_3d"    : pm25_roll_3d,
        "pm25_roll_7d"    : pm25_roll_7d,
        "pm25_roll_14d"   : pm25_roll_14d,
        "pm25_delta_1d"   : pm25_delta_1d,
    }

    # Aligner avec les features du modèle
    input_df = pd.DataFrame([input_dict])
    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[features]

    # ── Prédiction ────────────────────────────────────────────
    st.markdown("---")
    if st.button("🔮 Prédire le PM2.5 de demain", type="primary",
                 use_container_width=True):

        prediction = float(model.predict(input_df)[0])

        # Niveau d'alerte
        if prediction < 50:
            niveau = "🟢 BON"
            couleur = "green"
            conseil = "Qualité de l'air excellente. Activités extérieures sans restriction."
        elif prediction < 100:
            niveau = "🟡 MODÉRÉ"
            couleur = "orange"
            conseil = "Qualité acceptable. Les personnes sensibles doivent limiter les efforts prolongés."
        elif prediction < 150:
            niveau = "🟠 MAUVAIS"
            couleur = "darkorange"
            conseil = "Personnes sensibles : éviter les activités extérieures prolongées."
        elif prediction < 250:
            niveau = "🔴 TRÈS MAUVAIS"
            couleur = "red"
            conseil = "⚠️ Alerte pollution ! Limiter les sorties. Envisager restrictions de trafic."
        else:
            niveau = "🚨 DANGEREUX"
            couleur = "darkred"
            conseil = "🚨 URGENCE SANITAIRE. Fermeture écoles recommandée. Restrictions trafic obligatoires."

        # Affichage résultat
        col_r1, col_r2 = st.columns([1, 2])

        with col_r1:
            st.markdown(
                f"<div style='text-align:center; padding:30px; "
                f"border:3px solid {couleur}; border-radius:15px;'>"
                f"<h1 style='color:{couleur}'>{prediction:.1f}</h1>"
                f"<h3>µg/m³</h3>"
                f"<h2>{niveau}</h2>"
                f"</div>",
                unsafe_allow_html=True
            )

        with col_r2:
            # Jauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction,
                delta={"reference": stats["mean_pm25"],
                       "label": "vs moyenne historique"},
                gauge={
                    "axis": {"range": [0, 400]},
                    "bar" : {"color": couleur},
                    "steps": [
                        {"range": [0,   50],  "color": "#d4edda"},
                        {"range": [50,  100], "color": "#fff3cd"},
                        {"range": [100, 150], "color": "#fde8c8"},
                        {"range": [150, 250], "color": "#f8d7da"},
                        {"range": [250, 400], "color": "#721c24"},
                    ],
                    "threshold": {
                        "line" : {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": 150
                    }
                },
                title={"text": "PM2.5 prédit demain (µg/m³)"}
            ))
            fig_gauge.update_layout(height=300, margin=dict(t=40,b=0))
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.info(f"💡 **Conseil opérationnel** : {conseil}")

        # Intervalle de confiance approximatif
        st.caption(
            f"Intervalle estimé : [{max(0, prediction - stats['mae']):.1f}"
            f" — {prediction + stats['mae']:.1f}] µg/m³ "
            f"(± MAE = {stats['mae']:.1f} µg/m³)"
        )

# ══════════════════════════════════════════════════════════════
# PAGE 2 — Historique & Tendances
# ══════════════════════════════════════════════════════════════
elif page == "Historique & Tendances":

    st.title("📊 Historique de la Pollution PM2.5 — Beijing 2010–2014")

    # Filtre année
    years = sorted(df.index.year.unique())
    year_sel = st.multiselect("Années", years, default=years)
    df_sel = df[df.index.year.isin(year_sel)]

    # ── Série temporelle ──────────────────────────────────────
    daily = df_sel["pm25"].resample("D").mean()
    fig_ts = px.line(
        x=daily.index, y=daily.values,
        title="Évolution journalière du PM2.5",
        labels={"x": "Date", "y": "PM2.5 (µg/m³)"}
    )
    fig_ts.add_hline(y=150, line_dash="dash",
                     line_color="orange",
                     annotation_text="Seuil dangereux (150)")
    fig_ts.update_traces(line_color="#E74C3C", line_width=1)
    st.plotly_chart(fig_ts, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        # Saisonnalité mensuelle
        monthly = df_sel.groupby(df_sel.index.month)["pm25"].mean()
        mois = ["Jan","Fév","Mar","Avr","Mai","Jun",
                "Jul","Aoû","Sep","Oct","Nov","Déc"]
        fig_month = px.bar(
            x=[mois[i-1] for i in monthly.index],
            y=monthly.values,
            title="PM2.5 moyen par mois",
            labels={"x": "Mois", "y": "PM2.5 (µg/m³)"},
            color=monthly.values,
            color_continuous_scale="RdYlGn_r"
        )
        st.plotly_chart(fig_month, use_container_width=True)

    with col2:
        # Distribution par saison
        df_sel2 = df_sel.copy()
        df_sel2["saison"] = df_sel2.index.month.map({
            12:"Hiver", 1:"Hiver",  2:"Hiver",
            3:"Printemps", 4:"Printemps", 5:"Printemps",
            6:"Été",    7:"Été",    8:"Été",
            9:"Automne",10:"Automne",11:"Automne"
        })
        fig_box = px.box(
            df_sel2, x="saison", y="pm25",
            title="Distribution PM2.5 par saison",
            color="saison",
            category_orders={"saison": ["Hiver","Printemps",
                                         "Été","Automne"]},
            color_discrete_map={
                "Hiver":"#3498DB","Printemps":"#2ECC71",
                "Été":"#F39C12","Automne":"#E74C3C"
            }
        )
        fig_box.update_traces(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 3 — Analyse des performances
# ══════════════════════════════════════════════════════════════
elif page == "Analyse des performances":

    st.title("🔍 Performances du modèle sur 2014")

    # ✅ CORRIGÉ : suppression de test_df inutilisé

    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{stats['rmse']:.2f} µg/m³",
                help="Erreur quadratique moyenne")
    col2.metric("MAE",  f"{stats['mae']:.2f} µg/m³",
                help="Erreur absolue moyenne")
    col3.metric("R²",   f"{stats['r2']:.3f}",
                help="Variance expliquée")

    st.markdown("---")

    # Tableau des métriques comparatif
    st.subheader("📋 Comparaison des 3 modèles")
    df_metrics = pd.DataFrame({
        "Modèle" : ["Régression Linéaire",
                    "Random Forest", "XGBoost"],
        "RMSE"   : [52.00, 52.34, 51.50],
        "MAE"    : [38.88, 37.44, 36.76],
        "R²"     : [0.570, 0.565, 0.579],
    })
    st.dataframe(df_metrics, use_container_width=True,
                 hide_index=True)

    st.markdown("---")
    st.subheader("💡 Interprétation")
    st.markdown("""
    - **R² = 0.58** : le modèle explique 58% de la variance du PM2.5
      journalier — cohérent avec la littérature pour ce type de problème
    - **MAE = 37 µg/m³** : erreur absolue moyenne en conditions réelles
    - **Les 42% restants** correspondent à des facteurs non mesurés :
      feux agricoles, trafic réel, émissions industrielles ponctuelles
    """)

# ══════════════════════════════════════════════════════════════
# PAGE 4 — À propos — RAPPORT COMPLET
# ══════════════════════════════════════════════════════════════
elif page == "A propos du modele":

    # ── HERO ─────────────────────────────────────────────────
    st.markdown("""
    <div class="report-hero">
        <div class="report-hero-tag">Rapport technique &nbsp;·&nbsp; Smart City &nbsp;·&nbsp; 2025</div>
        <div class="report-hero-title">
            Prévision de la <em>qualité de l'air</em><br>à Beijing par Machine Learning
        </div>
        <div class="report-hero-subtitle">
            Modélisation prédictive du PM2.5 à horizon 24h à partir de données
            météorologiques et de séries temporelles de pollution. Une approche
            orientée aide à la décision urbaine.
        </div>
        <div class="report-hero-meta">
            <div class="report-hero-meta-item">
                <div class="report-hero-meta-label">Auteur</div>
                <div class="report-hero-meta-value">Abdoul Fataho NIAMPA</div>
            </div>
            <div class="report-hero-meta-item">
                <div class="report-hero-meta-label">Domaine</div>
                <div class="report-hero-meta-value">Data Science / Smart City</div>
            </div>
            <div class="report-hero-meta-item">
                <div class="report-hero-meta-label">Données</div>
                <div class="report-hero-meta-value">UCI Beijing PM2.5 Dataset</div>
            </div>
            <div class="report-hero-meta-item">
                <div class="report-hero-meta-label">Période</div>
                <div class="report-hero-meta-value">2010 – 2014</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION 1 — CONTEXTE ─────────────────────────────────
    st.markdown("""
    <div class="report-section-header">
        <span class="report-section-number">01</span>
        <span class="report-section-title">Contexte & Enjeux</span>
        <div class="report-section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    col_ctx1, col_ctx2 = st.columns([3, 2])
    with col_ctx1:
        st.markdown("""
        <div style="font-family:'DM Sans',sans-serif; font-size:15px; color:#334155; line-height:1.8;">
        <p>
        La pollution aux particules fines <strong style="color:#0f172a;">PM2.5</strong> constitue
        l'un des risques environnementaux les plus graves pour la santé publique en milieu urbain.
        À Beijing, ville de plus de 21 millions d'habitants, les épisodes de pollution intense
        affectent régulièrement la vie quotidienne, avec des pics atteignant
        <strong style="color:#dc2626;">500 µg/m³</strong> — soit 20 fois le seuil recommandé
        par l'OMS (25 µg/m³ sur 24h).
        </p>
        <p style="margin-top:16px;">
        Dans le cadre d'une vision <strong style="color:#0f172a;">Smart City</strong>, ce projet
        vise à fournir aux décideurs urbains un outil de prévision fiable à horizon J+1,
        permettant d'anticiper les alertes sanitaires, de planifier des restrictions de trafic
        et de communiquer en amont vers les populations sensibles.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col_ctx2:
        st.markdown("""
        <div style="background:#fef2f2; border:1px solid #fecaca; border-left:3px solid #dc2626;
                    border-radius:12px; padding:20px 22px; font-family:'DM Sans',sans-serif;">
            <div style="font-size:11px; color:#991b1b; text-transform:uppercase; letter-spacing:0.08em;
                        font-weight:600; margin-bottom:10px;">Seuils OMS — PM2.5</div>
            <div style="display:flex; justify-content:space-between; padding:8px 0;
                        border-bottom:1px solid #fecaca; font-size:13px;">
                <span style="color:#64748b;">Bon</span>
                <span style="color:#16a34a; font-weight:600;">&lt; 50 µg/m³</span>
            </div>
            <div style="display:flex; justify-content:space-between; padding:8px 0;
                        border-bottom:1px solid #fecaca; font-size:13px;">
                <span style="color:#64748b;">Modéré</span>
                <span style="color:#d97706; font-weight:600;">50 – 100 µg/m³</span>
            </div>
            <div style="display:flex; justify-content:space-between; padding:8px 0;
                        border-bottom:1px solid #fecaca; font-size:13px;">
                <span style="color:#64748b;">Mauvais</span>
                <span style="color:#ea580c; font-weight:600;">100 – 150 µg/m³</span>
            </div>
            <div style="display:flex; justify-content:space-between; padding:8px 0;
                        border-bottom:1px solid #fecaca; font-size:13px;">
                <span style="color:#64748b;">Très mauvais</span>
                <span style="color:#dc2626; font-weight:600;">150 – 250 µg/m³</span>
            </div>
            <div style="display:flex; justify-content:space-between; padding:8px 0;
                        font-size:13px;">
                <span style="color:#64748b;">Dangereux</span>
                <span style="color:#7f1d1d; font-weight:600;">&gt; 250 µg/m³</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── SECTION 2 — DONNÉES ──────────────────────────────────
    st.markdown("""
    <div class="report-section-header">
        <span class="report-section-number">02</span>
        <span class="report-section-title">Jeu de données</span>
        <div class="report-section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stat-cards-grid">
        <div class="stat-card">
            <div class="stat-card-accent" style="background:linear-gradient(90deg,#1D9E75,#34d399);"></div>
            <div class="stat-card-label">Observations horaires</div>
            <div class="stat-card-value">43 824</div>
            <div class="stat-card-unit">enregistrements</div>
            <div class="stat-card-desc">Données collectées sur 5 ans en continu (2010–2014) à la station US Embassy Beijing.</div>
        </div>
        <div class="stat-card">
            <div class="stat-card-accent" style="background:linear-gradient(90deg,#185FA5,#60a5fa);"></div>
            <div class="stat-card-label">PM2.5 moyen historique</div>
            <div class="stat-card-value">{stats['mean_pm25']:.0f}</div>
            <div class="stat-card-unit">µg/m³</div>
            <div class="stat-card-desc">Concentration moyenne observée sur l'ensemble de la période d'étude.</div>
        </div>
        <div class="stat-card">
            <div class="stat-card-accent" style="background:linear-gradient(90deg,#7c3aed,#a78bfa);"></div>
            <div class="stat-card-label">Features engineered</div>
            <div class="stat-card-value">39</div>
            <div class="stat-card-unit">variables</div>
            <div class="stat-card-desc">Variables construites à partir des séries temporelles brutes et des données météo.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-bottom:8px; font-family:'DM Sans',sans-serif;
                font-size:13px; color:#64748b;">Variables source disponibles :</div>
    <div class="feature-chips">
        <span class="feature-chip">pm2.5</span>
        <span class="feature-chip">TEMP</span>
        <span class="feature-chip">PRES</span>
        <span class="feature-chip">DEWP</span>
        <span class="feature-chip">Iws (vent)</span>
        <span class="feature-chip">Is (neige)</span>
        <span class="feature-chip">Ir (pluie)</span>
        <span class="feature-chip">cbwd (direction vent)</span>
        <span class="feature-chip">year / month / day / hour</span>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION 3 — MÉTHODOLOGIE ─────────────────────────────
    st.markdown("""
    <div class="report-section-header">
        <span class="report-section-number">03</span>
        <span class="report-section-title">Méthodologie</span>
        <div class="report-section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    col_m1, col_m2 = st.columns([1, 1])
    with col_m1:
        st.markdown("""
        <div class="timeline">
            <div class="timeline-item">
                <div class="timeline-dot"></div>
                <div class="timeline-step">Étape 01</div>
                <div class="timeline-title">Exploration & Analyse</div>
                <div class="timeline-desc">
                    Analyse de saisonnalité (décomposition STL), étude de la
                    stationnarité (test ADF), calcul des fonctions d'autocorrélation
                    (ACF / PACF) pour définir les lags pertinents.
                </div>
            </div>
            <div class="timeline-item">
                <div class="timeline-dot"></div>
                <div class="timeline-step">Étape 02</div>
                <div class="timeline-title">Feature Engineering</div>
                <div class="timeline-desc">
                    Construction de 39 features : lags temporels (1h, 6h, 12h, 24h,
                    1j, 2j, 7j), moyennes glissantes (3h, 12h, 24h, 3j, 7j, 14j),
                    variables météo dérivées (ressenti thermique, produits croisés),
                    encodages cycliques (saison, vent).
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown("""
        <div class="timeline">
            <div class="timeline-item">
                <div class="timeline-dot"></div>
                <div class="timeline-step">Étape 03</div>
                <div class="timeline-title">Modélisation & Validation</div>
                <div class="timeline-desc">
                    Split temporel strict : entraînement sur 2010–2013,
                    test sur 2014 (aucune fuite de données future).
                    3 modèles comparés : Régression Linéaire (baseline),
                    Random Forest (500 arbres), XGBoost.
                </div>
            </div>
            <div class="timeline-item">
                <div class="timeline-dot"></div>
                <div class="timeline-step">Étape 04</div>
                <div class="timeline-title">Interprétabilité</div>
                <div class="timeline-desc">
                    Analyse SHAP (SHapley Additive exPlanations) pour quantifier
                    la contribution de chaque feature. Feature importance globale
                    et locale pour explicabilité des prédictions.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── SECTION 4 — RÉSULTATS ────────────────────────────────
    st.markdown("""
    <div class="report-section-header">
        <span class="report-section-number">04</span>
        <span class="report-section-title">Résultats & Comparaison des modèles</span>
        <div class="report-section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <table class="model-table">
        <thead>
            <tr>
                <th>Modèle</th>
                <th>RMSE (µg/m³)</th>
                <th>MAE (µg/m³)</th>
                <th>R²</th>
                <th>Statut</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Régression Linéaire</td>
                <td>52.00</td>
                <td>38.88</td>
                <td>0.570</td>
                <td><span style="font-family:'JetBrains Mono',monospace;font-size:10px;
                    color:#94a3b8;background:#f1f5f9;padding:2px 8px;border-radius:20px;">Baseline</span></td>
            </tr>
            <tr class="winner">
                <td>Random Forest <span class="winner-badge">Retenu</span></td>
                <td style="color:#1D9E75;">{stats['rmse']:.2f}</td>
                <td style="color:#1D9E75;">{stats['mae']:.2f}</td>
                <td style="color:#1D9E75;">{stats['r2']:.3f}</td>
                <td><span style="font-family:'JetBrains Mono',monospace;font-size:10px;
                    color:#1D9E75;background:rgba(29,158,117,0.1);padding:2px 8px;
                    border-radius:20px;border:1px solid rgba(29,158,117,0.3);">Production</span></td>
            </tr>
            <tr>
                <td>XGBoost</td>
                <td>51.50</td>
                <td>36.76</td>
                <td>0.579</td>
                <td><span style="font-family:'JetBrains Mono',monospace;font-size:10px;
                    color:#94a3b8;background:#f1f5f9;padding:2px 8px;border-radius:20px;">Candidat</span></td>
            </tr>
        </tbody>
    </table>
    <div style="font-family:'DM Sans',sans-serif; font-size:12px; color:#94a3b8; margin-top:12px; font-style:italic;">
        Evaluation sur le jeu de test 2014 (split temporel strict — aucune donnée future utilisée à l'entraînement).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:28px; padding:24px 28px; background:#f8fafc; border-radius:14px;
                border:1px solid #e2e8f0; font-family:'DM Sans',sans-serif;">
        <div style="font-size:13px; font-weight:600; color:#0f172a; margin-bottom:14px;">
            Lecture des metriques
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px;">
            <div>
                <span style="color:#1D9E75; font-weight:700;">R² = 0.568</span>
                <span style="color:#64748b; font-size:13px;"> — Le modèle explique
                <strong>57% de la variance</strong> du PM2.5 journalier, cohérent
                avec la littérature scientifique pour ce type de prévision.</span>
            </div>
            <div>
                <span style="color:#185FA5; font-weight:700;">MAE = 37 µg/m³</span>
                <span style="color:#64748b; font-size:13px;"> — En conditions réelles,
                l'erreur absolue moyenne est de 37 µg/m³, soit environ
                <strong>38% de la valeur moyenne</strong> historique.</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION 5 — LIMITES ──────────────────────────────────
    st.markdown("""
    <div class="report-section-header">
        <span class="report-section-number">05</span>
        <span class="report-section-title">Limites & Perspectives</span>
        <div class="report-section-line"></div>
    </div>
    <div class="limit-grid">
        <div class="limit-card">
            <div class="limit-card-title">Portée géographique restreinte</div>
            <div class="limit-card-desc">Le modèle est entraîné exclusivement sur
            les données de Beijing. Sa généralisation à d'autres métropoles
            (Shanghai, Delhi, Paris) nécessiterait un réentraînement spécifique.</div>
        </div>
        <div class="limit-card">
            <div class="limit-card-title">Absence de données trafic & industrie</div>
            <div class="limit-card-desc">Les émissions liées au trafic routier
            et aux sites industriels ne sont pas captées. Ces facteurs expliquent
            une part significative des 42% de variance non modélisée.</div>
        </div>
        <div class="limit-card">
            <div class="limit-card-title">Horizon de prévision limité</div>
            <div class="limit-card-desc">Le modèle prédit uniquement à horizon
            J+1 (24h). Une extension à J+2 ou J+3 dégraderait sensiblement
            les performances sans revoir l'architecture de features.</div>
        </div>
        <div class="limit-card">
            <div class="limit-card-title">Feux agricoles & événements exceptionnels</div>
            <div class="limit-card-desc">Les pics extrêmes liés à des feux
            agricoles saisonniers ou des inversions thermiques sont difficiles
            à anticiper sans données de télédétection satellite.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION 6 — AUTEUR ───────────────────────────────────
    st.markdown("""
    <div class="author-card">
        <div class="author-avatar">AN</div>
        <div>
            <div class="author-name">Abdoul Fataho NIAMPA</div>
            <div class="author-role">Data Scientist &nbsp;·&nbsp; Projet Smart City Beijing</div>
            <a class="author-link"
               href="https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data"
               target="_blank">
               Source des données — UCI ML Repository →
            </a>
        </div>
        <div style="margin-left:auto; text-align:right;">
            <div style="font-family:'JetBrains Mono',monospace; font-size:10px;
                        color:#2d3748; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Version du modele
            </div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:13px; color:#1D9E75;">
                Random Forest v1.0
            </div>
            <div style="font-family:'DM Sans',sans-serif; font-size:12px; color:#4b5a6a; margin-top:4px;">
                scikit-learn — 500 estimateurs
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
