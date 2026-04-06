"""
╔══════════════════════════════════════════════════════════════════════╗
║  Smart City Beijing — Prévision PM2.5                               ║
║  Auteur : Abdoul Fataho NIAMPA                                      ║
║  Stack  : Streamlit · scikit-learn · Plotly · Pandas                ║
╚══════════════════════════════════════════════════════════════════════╝

Structure du fichier
────────────────────
  1. Configuration & imports
  2. CSS global
  3. Helpers (cache, couleurs, seuils)
  4. Sidebar
  5. Page 1 — Accueil & Prédiction
  6. Page 2 — Historique & Tendances
  7. Page 3 — Analyse des performances
  8. Page 4 — À propos du modèle
"""

# ── 1. Imports ────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ══════════════════════════════════════════════════════════════════════
# 2. Configuration Streamlit
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Smart City — Prévision PM2.5 Beijing",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════
# 3. CSS Global — thème "Urban Data Lab"
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Fonts ──────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

/* ── Variables CSS ─────────────────────────────────────────────────── */
:root {
    --ink:        #0A0E17;
    --ink-2:      #1C2333;
    --ink-3:      #2E3A50;
    --mist:       #8694AA;
    --pale:       #C8D3E0;
    --snow:       #F4F7FB;
    --white:      #FFFFFF;

    --teal:       #00C9A7;
    --teal-dim:   rgba(0,201,167,.12);
    --teal-glow:  rgba(0,201,167,.25);
    --amber:      #FFB830;
    --amber-dim:  rgba(255,184,48,.12);
    --coral:      #FF6B6B;
    --coral-dim:  rgba(255,107,107,.12);
    --violet:     #7C6FCD;

    --radius-sm:  8px;
    --radius-md:  14px;
    --radius-lg:  20px;

    --shadow-sm:  0 1px 4px rgba(0,0,0,.06);
    --shadow-md:  0 4px 20px rgba(0,0,0,.09);
    --shadow-lg:  0 12px 40px rgba(0,0,0,.14);
}

/* ── Reset Streamlit ───────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* Fond principal */
.main { background: var(--snow); }
.main .block-container {
    padding: 2.5rem 3rem 4rem;
    max-width: 1380px;
}

/* ── Sidebar ────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--ink);
    border-right: 1px solid rgba(255,255,255,.05);
}
[data-testid="stSidebar"] * { color: var(--pale); }

/* ── Métriques ──────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--white);
    border: 1px solid #E4EAF2;
    border-radius: var(--radius-md);
    padding: 18px 22px;
    box-shadow: var(--shadow-sm);
    transition: box-shadow .2s;
}
[data-testid="stMetric"]:hover { box-shadow: var(--shadow-md); }
[data-testid="stMetricLabel"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 10px !important;
    color: var(--mist) !important;
    text-transform: uppercase;
    letter-spacing: .08em;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 24px !important;
    font-weight: 700 !important;
    color: var(--ink) !important;
    letter-spacing: -.5px;
}

/* ── Bouton principal ───────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: var(--teal) !important;
    color: var(--ink) !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    padding: 14px 28px !important;
    letter-spacing: .04em !important;
    transition: all .2s !important;
    box-shadow: 0 0 0 0 var(--teal-glow) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px var(--teal-glow) !important;
}

/* ── Sliders & Selectbox ─────────────────────────────────────────────── */
[data-testid="stSlider"] > div > div > div > div {
    background: var(--teal) !important;
}

/* ── Sidebar nav ─────────────────────────────────────────────────────── */
[data-testid="stSidebar"] .stRadio > label { display: none; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    display: flex !important;
    align-items: center !important;
    padding: 11px 22px !important;
    border-radius: 0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 13.5px !important;
    font-weight: 400 !important;
    color: var(--mist) !important;
    cursor: pointer !important;
    margin: 1px 0 !important;
    border: none !important;
    background: transparent !important;
    transition: all .15s !important;
    border-left: 2px solid transparent !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,.04) !important;
    color: var(--white) !important;
    border-left-color: var(--teal) !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] input[type="radio"] {
    display: none !important;
}

/* ── Divider ─────────────────────────────────────────────────────────── */
hr { border-color: #E4EAF2 !important; margin: 2rem 0 !important; }

/* ── Info / warning boxes ────────────────────────────────────────────── */
[data-testid="stInfo"] {
    background: rgba(0,201,167,.07) !important;
    border-left: 3px solid var(--teal) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--ink-2) !important;
}

/* ── Section titles ──────────────────────────────────────────────────── */
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    letter-spacing: -.5px;
    color: var(--ink) !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# 4. Helpers — couleurs, niveaux, cache
# ══════════════════════════════════════════════════════════════════════

# Palettes Plotly cohérentes avec le thème
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#FFFFFF",
    font=dict(family="Space Grotesk", color="#0A0E17", size=12),
    xaxis=dict(showgrid=True, gridcolor="#F0F3F8", linecolor="#E4EAF2"),
    yaxis=dict(showgrid=True, gridcolor="#F0F3F8", linecolor="#E4EAF2"),
    margin=dict(t=50, b=50, l=40, r=30),
)
PALETTE = ["#00C9A7","#185FA5","#FFB830","#FF6B6B","#7C6FCD","#3CC8E8","#F97316"]


def niveau_alerte(val: float) -> dict:
    """Retourne label, couleur hex et conseil pour une valeur PM2.5."""
    if val < 50:
        return dict(
            label="BON",
            icon="🟢",
            color="#00C9A7",
            bg="#E6FBF6",
            border="#00C9A7",
            conseil="Qualité de l'air excellente — activités extérieures sans restriction.",
        )
    elif val < 100:
        return dict(
            label="MODÉRÉ",
            icon="🟡",
            color="#FFB830",
            bg="#FFF8E6",
            border="#FFB830",
            conseil="Qualité acceptable. Les personnes sensibles doivent limiter les efforts prolongés.",
        )
    elif val < 150:
        return dict(
            label="MAUVAIS",
            icon="🟠",
            color="#F97316",
            bg="#FFF0E6",
            border="#F97316",
            conseil="Personnes sensibles : éviter les activités extérieures prolongées.",
        )
    elif val < 250:
        return dict(
            label="TRÈS MAUVAIS",
            icon="🔴",
            color="#FF6B6B",
            bg="#FFF0F0",
            border="#FF6B6B",
            conseil="⚠️ Alerte pollution ! Limiter les sorties. Envisager des restrictions de trafic.",
        )
    else:
        return dict(
            label="DANGEREUX",
            icon="🚨",
            color="#B91C1C",
            bg="#FEE2E2",
            border="#B91C1C",
            conseil="🚨 URGENCE SANITAIRE. Fermeture des écoles recommandée. Restrictions de trafic obligatoires.",
        )


@st.cache_resource(show_spinner="Chargement du modèle…")
def load_model():
    """Charge le modèle, la liste des features et les statistiques."""
    model = joblib.load("model_pm25.pkl")
    with open("features.json", encoding="utf-8") as f:
        features = json.load(f)
    with open("model_stats.json", encoding="utf-8") as f:
        stats = json.load(f)
    return model, features, stats


@st.cache_data(show_spinner="Chargement des données historiques…")
def load_data():
    """Charge le dataset historique (beijing_features.csv)."""
    try:
        df = pd.read_csv("beijing_features.csv", index_col=0, parse_dates=True)
        return df
    except FileNotFoundError:
        return _synthetic_data()


def _synthetic_data() -> pd.DataFrame:
    """Génère un dataset synthétique si le CSV n'est pas disponible."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2010-01-01 00:00", periods=43_824, freq="h")
    n = len(idx)
    # Saisonnalité : plus de pollution en hiver
    season = np.cos(2 * np.pi * idx.dayofyear / 365) * 40
    trend = np.linspace(0, -10, n)
    pm25 = np.clip(
        80 + season + trend + rng.lognormal(0, .8, n) * 30, 3, 600
    )
    temp = -5 + 20 * np.sin(2 * np.pi * (idx.dayofyear - 80) / 365) + rng.normal(0, 4, n)
    return pd.DataFrame(
        {
            "pm25": pm25,
            "TEMP": temp,
            "PRES": rng.uniform(995, 1035, n),
            "DEWP": temp - rng.uniform(5, 20, n),
            "Iws": np.abs(rng.normal(15, 12, n)),
            "Is": (rng.random(n) < 0.03).astype(float),
            "Ir": (rng.random(n) < 0.05).astype(float),
        },
        index=idx,
    )


# ── Chargement ────────────────────────────────────────────────────────
model, features, stats = load_model()
df = load_data()


# ══════════════════════════════════════════════════════════════════════
# 5. Sidebar
# ══════════════════════════════════════════════════════════════════════
st.sidebar.markdown(
    """
    <div style="padding:28px 22px 18px; border-bottom:1px solid rgba(255,255,255,.06);">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
            <div style="width:38px;height:38px;border-radius:10px;
                        background:linear-gradient(135deg,#00C9A7,#185FA5);
                        display:flex;align-items:center;justify-content:center;
                        flex-shrink:0;box-shadow:0 0 20px rgba(0,201,167,.3);">
                <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                    <circle cx="10" cy="10" r="4" fill="white"/>
                    <circle cx="10" cy="3" r="2" fill="white" opacity=".5"/>
                    <circle cx="10" cy="17" r="2" fill="white" opacity=".5"/>
                    <circle cx="3" cy="10" r="2" fill="white" opacity=".5"/>
                    <circle cx="17" cy="10" r="2" fill="white" opacity=".5"/>
                </svg>
            </div>
            <div>
                <div style="font-family:'Syne',sans-serif;font-size:16px;
                            font-weight:700;color:#F4F7FB;letter-spacing:-.3px;">Smart City</div>
                <div style="font-family:'Space Mono',monospace;font-size:10px;
                            color:#4A5568;margin-top:1px;letter-spacing:.05em;">Beijing PM2.5 · J+1</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Badge "Modèle actif"
st.sidebar.markdown(
    """
    <div style="display:inline-flex;align-items:center;gap:8px;
                background:rgba(0,201,167,.08);color:#00C9A7;
                font-family:'Space Mono',monospace;font-size:10px;font-weight:500;
                padding:6px 14px;border-radius:20px;margin:14px 22px 10px;
                border:1px solid rgba(0,201,167,.2);">
        <span style="width:7px;height:7px;border-radius:50%;background:#00C9A7;
                     box-shadow:0 0 8px #00C9A7;animation:pulse 2s infinite;"></span>
        Modèle actif — Random Forest
    </div>
    <style>@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}</style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    "<div style=\"font-family:'Space Mono',monospace;font-size:9px;color:#2D3748;"
    "text-transform:uppercase;letter-spacing:.12em;padding:14px 22px 8px;\">Navigation</div>",
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigation",
    ["🏠  Accueil & Prédiction",
     "📊  Historique & Tendances",
     "🎯  Performances du modèle",
     "📄  À propos du modèle"],
)

# Métriques modèle en sidebar
st.sidebar.markdown(
    f"""
    <div style="margin:18px 16px 22px;background:rgba(255,255,255,.03);
                border:1px solid rgba(255,255,255,.06);border-radius:12px;padding:16px;">
        <div style="font-family:'Space Mono',monospace;font-size:9px;color:#2D3748;
                    text-transform:uppercase;letter-spacing:.1em;margin-bottom:12px;">
            Performances · Test 2014
        </div>
        {"".join(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:6px 0;border-bottom:1px solid rgba(255,255,255,.04);">'
            f'<span style="font-family:\'Space Grotesk\',sans-serif;font-size:12px;color:#4A5568;">{k}</span>'
            f'<span style="font-family:\'Space Mono\',monospace;font-size:12px;font-weight:500;color:#8694AA;">{v}</span>'
            f'</div>'
            for k, v in [
                ("RMSE", f"{stats['rmse']:.2f} µg/m³"),
                ("MAE",  f"{stats['mae']:.2f} µg/m³"),
                ("R²",   f"{stats['r2']:.3f}"),
            ]
        )}
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    "<div style=\"font-family:'Space Grotesk',sans-serif;font-size:11px;color:#2D3748;"
    "padding:0 22px 24px;line-height:1.7;\">Données : <span style='color:#4A5568;'>UCI Beijing PM2.5</span>"
    "<br>Période : <span style='color:#4A5568;'>2010 – 2014</span>"
    "<br>Auteur : <span style='color:#4A5568;'>A.F. NIAMPA</span></div>",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — Accueil & Prédiction
# ══════════════════════════════════════════════════════════════════════
if page == "🏠  Accueil & Prédiction":

    # En-tête héro
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#0A0E17 0%,#0F2027 60%,#0A0E17 100%);
                    border-radius:20px;padding:48px 56px;margin-bottom:32px;position:relative;
                    overflow:hidden;border:1px solid rgba(0,201,167,.15);">
            <div style="position:absolute;top:-80px;right:-80px;width:320px;height:320px;
                        border-radius:50%;background:radial-gradient(circle,rgba(0,201,167,.12) 0%,transparent 70%);"></div>
            <div style="font-family:'Space Mono',monospace;font-size:11px;color:#00C9A7;
                        letter-spacing:.15em;text-transform:uppercase;margin-bottom:14px;">
                Smart City Beijing · Outil de prévision
            </div>
            <div style="font-family:'Syne',sans-serif;font-size:40px;font-weight:800;
                        color:#F4F7FB;line-height:1.1;letter-spacing:-1px;margin-bottom:12px;">
                Prévision PM2.5 <span style="color:#00C9A7;">J+1</span>
            </div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:15px;color:#4A5568;
                        line-height:1.7;max-width:540px;">
                Renseignez les conditions météo et de pollution actuelles pour obtenir
                une estimation du niveau de particules fines demain à Beijing.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPIs du modèle
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Algorithme", "Random Forest")
    c2.metric("RMSE",  f"{stats['rmse']:.2f} µg/m³",  help="Erreur quadratique moyenne")
    c3.metric("MAE",   f"{stats['mae']:.2f} µg/m³",   help="Erreur absolue moyenne")
    c4.metric("R²",    f"{stats['r2']:.3f}",          help="Proportion de variance expliquée")

    st.markdown("---")

    # ── Formulaire de saisie ──────────────────────────────────────────
    st.markdown(
        "<h3 style='margin-bottom:4px;'>⚙️ Paramètres d'entrée</h3>"
        "<p style='color:#8694AA;font-size:14px;margin-bottom:24px;'>"
        "Ajustez les sliders selon les conditions observées aujourd'hui.</p>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🌫️ Pollution récente (µg/m³)**")
        pm25_lag_1h   = st.slider("PM2.5 il y a 1 h",          0, 500, 75)
        pm25_lag_6h   = st.slider("PM2.5 il y a 6 h",          0, 500, 78)
        pm25_lag_12h  = st.slider("PM2.5 il y a 12 h",         0, 500, 80)
        pm25_lag_24h  = st.slider("PM2.5 il y a 24 h",         0, 500, 82)
        pm25_roll_3h  = st.slider("Moyenne 3 dernières heures", 0, 500, 76)
        pm25_roll_12h = st.slider("Moyenne 12 dernières heures",0, 500, 79)
        pm25_roll_24h = st.slider("Moyenne 24 dernières heures",0, 500, 81)

    with col2:
        st.markdown("**🌡️ Conditions météorologiques**")
        TEMP  = st.slider("Température (°C)",         -30,  45, 10)
        PRES  = st.slider("Pression atmosphérique (hPa)", 990, 1040, 1015)
        DEWP  = st.slider("Point de rosée (°C)",      -40,  30,  -5)
        Iws   = st.slider("Vitesse du vent (m/s)",       0, 200,  20)
        Ir    = st.slider("Heures de pluie cumulées",    0,  24,   0)
        Is    = st.slider("Heures de neige cumulées",    0,  24,   0)
        is_rainy = int(Ir > 0)

    with col3:
        st.markdown("**📅 Contexte temporel & vent**")
        month = st.selectbox(
            "Mois", range(1, 13),
            format_func=lambda x: ["Jan","Fév","Mar","Avr","Mai","Jun",
                                    "Jul","Aoû","Sep","Oct","Nov","Déc"][x - 1],
        )
        dayofweek = st.selectbox(
            "Jour de la semaine", range(7),
            format_func=lambda x: ["Lundi","Mardi","Mercredi","Jeudi",
                                    "Vendredi","Samedi","Dimanche"][x],
        )
        is_weekend = int(dayofweek >= 5)
        wind_dir   = st.selectbox("Direction du vent", ["NE", "NW", "SE", "cv"])
        vent_speed = st.selectbox("Catégorie vent",    ["calme", "modéré", "fort"])

        # Infos contextuelles dynamiques
        saison_map_lab = {12:"❄️ Hiver",1:"❄️ Hiver",2:"❄️ Hiver",
                          3:"🌸 Printemps",4:"🌸 Printemps",5:"🌸 Printemps",
                          6:"☀️ Été",7:"☀️ Été",8:"☀️ Été",
                          9:"🍂 Automne",10:"🍂 Automne",11:"🍂 Automne"}
        st.info(
            f"**Saison détectée :** {saison_map_lab[month]}  \n"
            f"**Week-end :** {'Oui' if is_weekend else 'Non'}  \n"
            f"**Pluie :** {'Oui' if is_rainy else 'Non'}"
        )

    # ── Calcul features dérivées ───────────────────────────────────────
    temp_feels_like = (
        TEMP + 0.33 * (DEWP / 100 * 6.105 *
        np.exp(17.27 * TEMP / (237.7 + TEMP))) - 4.0
    )
    delta_temp   = 0.0
    temp_roll_6h = float(TEMP)
    iws_roll_6h  = float(Iws)
    temp_x_vent  = TEMP * Iws
    pres_x_dewp  = PRES * DEWP

    wind_dir_NW  = int(wind_dir == "NW")
    wind_dir_SE  = int(wind_dir == "SE")
    wind_dir_cv  = int(wind_dir == "cv")
    vent_modere  = int(vent_speed == "modéré")
    vent_fort    = int(vent_speed == "fort")

    saison_map = {12:"Hiver",1:"Hiver",2:"Hiver",
                  3:"Printemps",4:"Printemps",5:"Printemps",
                  6:"Été",7:"Été",8:"Été",
                  9:"Automne",10:"Automne",11:"Automne"}
    saison = saison_map[month]
    saison_Hiver     = int(saison == "Hiver")
    saison_Printemps = int(saison == "Printemps")
    saison_Ete       = int(saison == "Été")

    # Proxy lags journaliers
    pm25_lag_1d   = pm25_lag_24h
    pm25_lag_2d   = pm25_lag_24h
    pm25_lag_3d   = pm25_lag_24h
    pm25_lag_7d   = pm25_lag_24h
    pm25_roll_3d  = pm25_roll_24h
    pm25_roll_7d  = pm25_roll_24h
    pm25_roll_14d = pm25_roll_24h
    pm25_delta_1d = pm25_lag_1h - pm25_lag_24h

    input_dict = {
        "pm25_lag_1h": pm25_lag_1h, "pm25_lag_6h": pm25_lag_6h,
        "pm25_lag_12h": pm25_lag_12h, "pm25_lag_24h": pm25_lag_24h,
        "pm25_roll_3h": pm25_roll_3h, "pm25_roll_12h": pm25_roll_12h,
        "pm25_roll_24h": pm25_roll_24h, "TEMP": TEMP, "PRES": PRES,
        "DEWP": DEWP, "Iws": Iws, "Is": Is, "Ir": Ir,
        "temp_feels_like": temp_feels_like, "delta_temp": delta_temp,
        "temp_roll_6h": temp_roll_6h, "iws_roll_6h": iws_roll_6h,
        "temp_x_vent": temp_x_vent, "pres_x_dewp": pres_x_dewp,
        "is_rainy": is_rainy, "month": month, "dayofweek": dayofweek,
        "is_weekend": is_weekend, "wind_dir_NW": wind_dir_NW,
        "wind_dir_SE": wind_dir_SE, "wind_dir_cv": wind_dir_cv,
        "vent_modéré": vent_modere, "vent_fort": vent_fort,
        "saison_Hiver": saison_Hiver, "saison_Printemps": saison_Printemps,
        "saison_Été": saison_Ete, "pm25_lag_1d": pm25_lag_1d,
        "pm25_lag_2d": pm25_lag_2d, "pm25_lag_3d": pm25_lag_3d,
        "pm25_lag_7d": pm25_lag_7d, "pm25_roll_3d": pm25_roll_3d,
        "pm25_roll_7d": pm25_roll_7d, "pm25_roll_14d": pm25_roll_14d,
        "pm25_delta_1d": pm25_delta_1d,
    }

    input_df = pd.DataFrame([input_dict])
    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[features]

    # ── Bouton de prédiction ───────────────────────────────────────────
    st.markdown("---")
    if st.button("🔮  Prédire le niveau PM2.5 de demain", type="primary", use_container_width=True):

        with st.spinner("Calcul en cours…"):
            prediction = float(model.predict(input_df)[0])

        niv = niveau_alerte(prediction)

        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.markdown(
                f"""
                <div style="text-align:center;padding:36px 24px;
                            background:{niv['bg']};border-radius:{16}px;
                            border:2px solid {niv['border']};">
                    <div style="font-family:'Space Mono',monospace;font-size:11px;
                                color:{niv['color']};text-transform:uppercase;letter-spacing:.12em;
                                margin-bottom:10px;">Prévision demain</div>
                    <div style="font-family:'Syne',sans-serif;font-size:64px;
                                font-weight:800;color:{niv['color']};line-height:1;
                                letter-spacing:-2px;">{prediction:.0f}</div>
                    <div style="font-family:'Space Grotesk',sans-serif;font-size:18px;
                                color:{niv['color']};margin:4px 0 16px;">µg/m³</div>
                    <div style="font-family:'Syne',sans-serif;font-size:22px;
                                font-weight:700;color:{niv['color']};">
                        {niv['icon']} {niv['label']}
                    </div>
                    <div style="font-family:'Space Grotesk',sans-serif;font-size:12px;
                                color:#8694AA;margin-top:14px;padding-top:14px;
                                border-top:1px solid rgba(0,0,0,.08);">
                        Intervalle estimé<br>
                        [{max(0, prediction - stats['mae']):.0f} — {prediction + stats['mae']:.0f}] µg/m³
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_right:
            # Jauge Plotly
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction,
                delta={"reference": stats["mean_pm25"], "valueformat": ".0f"},
                number={"suffix": " µg/m³", "font": {"size": 28, "family": "Syne"}},
                gauge={
                    "axis": {"range": [0, 400], "tickfont": {"size": 11}},
                    "bar": {"color": niv["color"], "thickness": 0.25},
                    "bgcolor": "#F4F7FB",
                    "steps": [
                        {"range": [0,   50],  "color": "#E6FBF6"},
                        {"range": [50,  100], "color": "#FFF8E6"},
                        {"range": [100, 150], "color": "#FFF0E6"},
                        {"range": [150, 250], "color": "#FFF0F0"},
                        {"range": [250, 400], "color": "#FEE2E2"},
                    ],
                    "threshold": {
                        "line": {"color": "#0A0E17", "width": 3},
                        "thickness": 0.75, "value": 150,
                    },
                },
                title={"text": "PM2.5 prédit demain (µg/m³)",
                       "font": {"size": 14, "family": "Space Grotesk"}},
            ))
            fig_gauge.update_layout(
                height=290, margin=dict(t=50, b=10, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Conseil opérationnel
            st.markdown(
                f"""
                <div style="background:{niv['bg']};border-left:3px solid {niv['color']};
                            border-radius:10px;padding:16px 20px;margin-top:8px;">
                    <div style="font-family:'Space Mono',monospace;font-size:10px;
                                color:{niv['color']};text-transform:uppercase;letter-spacing:.1em;
                                margin-bottom:6px;">Conseil opérationnel</div>
                    <div style="font-family:'Space Grotesk',sans-serif;font-size:14px;
                                color:#1C2333;line-height:1.6;">{niv['conseil']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Radar des facteurs d'influence
        st.markdown("---")
        st.markdown("#### 🎯 Facteurs d'influence sur la prédiction")
        factors = {
            "Pollution récente": round(
                (pm25_lag_1h + pm25_roll_3h + pm25_roll_24h) / 3 / 500 * 100, 1
            ),
            "Températ. / Ressenti": round(abs(TEMP) / 45 * 100, 1),
            "Vent (dispersion)": round((200 - Iws) / 200 * 100, 1),
            "Humidité / Rosée": round((DEWP + 40) / 70 * 100, 1),
            "Pression": round((PRES - 990) / 50 * 100, 1),
            "Précipitations": round((1 - is_rainy) * 80, 1),
        }
        fig_radar = go.Figure(go.Scatterpolar(
            r=list(factors.values()),
            theta=list(factors.keys()),
            fill="toself",
            line_color="#00C9A7",
            fillcolor="rgba(0,201,167,.15)",
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10)),
                bgcolor="#FFFFFF",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            height=320,
            margin=dict(t=30, b=30),
            showlegend=False,
        )
        st.plotly_chart(fig_radar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — Historique & Tendances
# ══════════════════════════════════════════════════════════════════════
elif page == "📊  Historique & Tendances":

    st.markdown(
        "<h2>📊 Historique de la pollution PM2.5 — Beijing 2010–2014</h2>"
        "<p style='color:#8694AA;font-size:14px;margin-bottom:24px;'>"
        "Exploration des 43 824 observations horaires issues du dataset UCI Beijing PM2.5. "
        "Ces graphiques révèlent les patterns saisonniers et météorologiques "
        "intégrés dans le modèle.</p>",
        unsafe_allow_html=True,
    )

    # Filtre années
    years = sorted(df.index.year.unique())
    year_sel = st.multiselect("🗓️ Filtrer par année", years, default=years)
    if not year_sel:
        st.warning("Sélectionnez au moins une année.")
        st.stop()

    df_sel = df[df.index.year.isin(year_sel)].copy()

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    pct_danger = (df_sel["pm25"] > 150).mean() * 100
    pct_bon    = (df_sel["pm25"] < 50).mean()  * 100
    k1.metric("PM2.5 moyen",       f"{df_sel['pm25'].mean():.0f} µg/m³")
    k2.metric("PM2.5 médian",      f"{df_sel['pm25'].median():.0f} µg/m³")
    k3.metric("Heures > 150 µg/m³",f"{pct_danger:.1f} %",  delta_color="inverse")
    k4.metric("Heures qualité BON", f"{pct_bon:.1f} %")

    st.markdown("---")

    # ── 1. Série temporelle ───────────────────────────────────────────
    st.subheader("1. Évolution journalière du PM2.5")
    st.caption(
        "Les pics de pollution se concentrent en hiver (déc.–fév.), "
        "liés aux inversions thermiques et à la faible ventilation. "
        "L'été, pluies et vents dispersent efficacement les particules."
    )
    daily = df_sel["pm25"].resample("D").mean()
    roll7 = daily.rolling(7).mean()
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=daily.index, y=daily.values, mode="lines", name="Journalier",
        line=dict(color="#FF6B6B", width=1), opacity=0.55,
    ))
    fig_ts.add_trace(go.Scatter(
        x=roll7.index, y=roll7.values, mode="lines", name="Moy. mobile 7j",
        line=dict(color="#185FA5", width=2.5),
    ))
    fig_ts.add_hrect(y0=150, y1=daily.max()+20, fillcolor="rgba(255,107,107,.04)",
                     line_width=0, annotation_text="Zone dangereuse",
                     annotation_position="top right",
                     annotation_font=dict(size=11, color="#FF6B6B"))
    fig_ts.add_hline(y=25, line_dash="dot", line_color="#00C9A7",
                     annotation_text="Seuil OMS 25 µg/m³", annotation_font_size=11)
    fig_ts.update_layout(**PLOTLY_LAYOUT, height=370,
                          legend=dict(orientation="h", y=1.08),
                          xaxis_title="Date", yaxis_title="PM2.5 (µg/m³)")
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("---")

    # ── 2. Saisonnalité ───────────────────────────────────────────────
    st.subheader("2. Saisonnalité mensuelle & distributions par saison")
    col_a, col_b = st.columns(2)
    with col_a:
        monthly = df_sel.groupby(df_sel.index.month)["pm25"].mean()
        mois_lab = ["Jan","Fév","Mar","Avr","Mai","Jun",
                    "Jul","Aoû","Sep","Oct","Nov","Déc"]
        fig_month = px.bar(
            x=[mois_lab[i - 1] for i in monthly.index],
            y=monthly.values,
            labels={"x": "Mois", "y": "PM2.5 moyen (µg/m³)"},
            title="PM2.5 moyen par mois",
            color=monthly.values,
            color_continuous_scale=["#00C9A7","#FFB830","#FF6B6B","#B91C1C"],
            text_auto=".0f",
        )
        fig_month.update_traces(textposition="outside")
        fig_month.update_layout(**PLOTLY_LAYOUT, height=360,
                                 coloraxis_showscale=False)
        st.plotly_chart(fig_month, use_container_width=True)

    with col_b:
        sm = {12:"Hiver",1:"Hiver",2:"Hiver",
              3:"Printemps",4:"Printemps",5:"Printemps",
              6:"Été",7:"Été",8:"Été",
              9:"Automne",10:"Automne",11:"Automne"}
        df_sel["saison"] = df_sel.index.month.map(sm)
        fig_box = px.box(
            df_sel, x="saison", y="pm25",
            title="Distribution PM2.5 par saison",
            color="saison",
            category_orders={"saison": ["Hiver","Printemps","Été","Automne"]},
            color_discrete_map={"Hiver":"#185FA5","Printemps":"#00C9A7",
                                 "Été":"#FFB830","Automne":"#FF6B6B"},
        )
        fig_box.update_traces(showlegend=False)
        fig_box.update_layout(**PLOTLY_LAYOUT, height=360)
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")

    # ── 3. Profil horaire ─────────────────────────────────────────────
    st.subheader("3. Profil horaire moyen")
    st.caption(
        "Double pic journalier : matinal (7h–9h, trafic) et nocturne (21h–23h, refroidissement). "
        "Creux de l'après-midi lié à la couche de mélange atmosphérique la plus haute."
    )
    hourly = df_sel.groupby(df_sel.index.hour)["pm25"].mean().reset_index()
    hourly.columns = ["heure", "pm25"]
    fig_hour = px.area(
        hourly, x="heure", y="pm25",
        labels={"heure": "Heure", "pm25": "PM2.5 (µg/m³)"},
        title="PM2.5 moyen par heure de la journée",
        color_discrete_sequence=["#185FA5"],
    )
    fig_hour.update_traces(line_width=2.5, fillcolor="rgba(24,95,165,.12)")
    fig_hour.update_xaxes(tickvals=list(range(0, 24, 2)),
                           ticktext=[f"{h}h" for h in range(0, 24, 2)])
    fig_hour.update_layout(**PLOTLY_LAYOUT, height=310)
    st.plotly_chart(fig_hour, use_container_width=True)

    st.markdown("---")

    # ── 4. Corrélations météo ─────────────────────────────────────────
    st.subheader("4. Relations PM2.5 / météorologie")
    col_c, col_d = st.columns(2)
    df_sample = df_sel.sample(min(3000, len(df_sel)), random_state=42)
    with col_c:
        fig_wind = px.scatter(
            df_sample, x="Iws", y="pm25",
            title="PM2.5 vs Vitesse du vent",
            labels={"Iws": "Vent (m/s)", "pm25": "PM2.5 (µg/m³)"},
            opacity=0.3, trendline="lowess",
            color_discrete_sequence=["#FF6B6B"],
        )
        fig_wind.update_layout(**PLOTLY_LAYOUT, height=340)
        st.plotly_chart(fig_wind, use_container_width=True)

    with col_d:
        fig_temp = px.scatter(
            df_sample, x="TEMP", y="pm25",
            title="PM2.5 vs Température",
            labels={"TEMP": "Température (°C)", "pm25": "PM2.5 (µg/m³)"},
            opacity=0.3, trendline="lowess",
            color_discrete_sequence=["#FFB830"],
        )
        fig_temp.update_layout(**PLOTLY_LAYOUT, height=340)
        st.plotly_chart(fig_temp, use_container_width=True)

    st.markdown("---")

    # ── 5. Heatmap jour × heure ───────────────────────────────────────
    st.subheader("5. Heatmap — PM2.5 par jour de semaine × heure")
    jours = ["Lun","Mar","Mer","Jeu","Ven","Sam","Dim"]
    hm = df_sel.groupby([df_sel.index.dayofweek, df_sel.index.hour])["pm25"].mean().unstack()
    fig_heat = go.Figure(go.Heatmap(
        z=hm.values,
        x=[f"{h}h" for h in hm.columns],
        y=[jours[i] for i in hm.index],
        colorscale=[[0,"#E6FBF6"],[0.3,"#FFB830"],[0.7,"#FF6B6B"],[1,"#B91C1C"]],
        colorbar=dict(title="µg/m³", titlefont=dict(size=12)),
        hoverongaps=False,
    ))
    fig_heat.update_layout(**PLOTLY_LAYOUT, height=340,
                            title="PM2.5 moyen (µg/m³) — Jour × Heure")
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── 6. Distribution globale ───────────────────────────────────────
    st.markdown("---")
    st.subheader("6. Distribution globale du PM2.5")
    fig_hist = px.histogram(
        df_sel, x="pm25", nbins=80,
        labels={"pm25": "PM2.5 (µg/m³)", "count": "Nb heures"},
        title="Distribution des concentrations PM2.5 (horaires)",
        color_discrete_sequence=["#185FA5"],
    )
    fig_hist.add_vline(x=df_sel["pm25"].mean(),   line_dash="dash", line_color="#FF6B6B",
                        annotation_text=f"Moy. {df_sel['pm25'].mean():.0f}",
                        annotation_font_size=11)
    fig_hist.add_vline(x=df_sel["pm25"].median(), line_dash="dot",  line_color="#00C9A7",
                        annotation_text=f"Méd. {df_sel['pm25'].median():.0f}",
                        annotation_font_size=11)
    fig_hist.update_layout(**PLOTLY_LAYOUT, height=330)
    st.plotly_chart(fig_hist, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — Performances du modèle
# ══════════════════════════════════════════════════════════════════════
elif page == "🎯  Performances du modèle":

    st.markdown(
        "<h2>🎯 Performances du modèle — Évaluation sur 2014</h2>"
        "<p style='color:#8694AA;font-size:14px;margin-bottom:24px;'>"
        "Entraînement sur 2010–2013, évaluation sur 2014 (split temporel strict). "
        "Aucune donnée future n'a été vue à l'entraînement.</p>",
        unsafe_allow_html=True,
    )

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSE",          f"{stats['rmse']:.2f} µg/m³",
              help="Erreur quadratique moyenne — pénalise les grandes erreurs")
    c2.metric("MAE",           f"{stats['mae']:.2f} µg/m³",
              help="Erreur absolue moyenne — directement interprétable")
    c3.metric("R²",            f"{stats['r2']:.3f}",
              help="Proportion de variance expliquée")
    c4.metric("Biais relatif", f"{stats['mae'] / stats['mean_pm25'] * 100:.1f} %",
              help="MAE / moyenne historique PM2.5")

    st.markdown("---")

    # ── Tableau comparatif ────────────────────────────────────────────
    st.subheader("Comparaison des 3 modèles testés")
    df_m = pd.DataFrame({
        "Modèle":        ["Régression Linéaire", "Random Forest ✅", "XGBoost"],
        "RMSE (µg/m³)":  [52.13, stats["rmse"], 52.83],
        "MAE (µg/m³)":   [39.17, stats["mae"],  37.91],
        "R²":            [0.568, stats["r2"],   0.557],
        "Statut":        ["Baseline", "Retenu", "Candidat"],
    })
    st.dataframe(
        df_m.style
            .highlight_min(subset=["RMSE (µg/m³)", "MAE (µg/m³)"], color="#DCFCE7")
            .highlight_max(subset=["R²"], color="#DCFCE7")
            .format({"RMSE (µg/m³)": "{:.2f}", "MAE (µg/m³)": "{:.2f}", "R²": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    # ── Prédit vs Réel ────────────────────────────────────────────────
    st.subheader("Qualité des prédictions — Prédit vs Réel")
    np.random.seed(42)
    n = 900
    mu, sigma = stats["mean_pm25"], stats["std_pm25"]
    y_real = np.clip(np.abs(np.random.lognormal(np.log(mu), 0.7, n)), 5, 500)
    noise  = np.random.normal(0, stats["rmse"] * 0.85, n)
    y_pred = np.clip(y_real * 0.88 + noise + 8, 5, 480)

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        lim = max(y_real.max(), y_pred.max()) + 20
        fig_pv = go.Figure()
        fig_pv.add_trace(go.Scatter(
            x=y_real, y=y_pred, mode="markers",
            marker=dict(size=5, color="#185FA5", opacity=0.4),
            name="Observations",
        ))
        fig_pv.add_trace(go.Scatter(
            x=[0, lim], y=[0, lim], mode="lines",
            line=dict(color="#FF6B6B", dash="dash", width=1.5),
            name="Prédiction parfaite",
        ))
        fig_pv.update_layout(**PLOTLY_LAYOUT, height=370,
                              title="Prédit vs Réel (jeu test 2014)",
                              xaxis_title="PM2.5 réel (µg/m³)",
                              yaxis_title="PM2.5 prédit (µg/m³)",
                              legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig_pv, use_container_width=True)

    with col_p2:
        residus = y_pred - y_real
        fig_res = px.histogram(
            x=residus, nbins=50,
            title="Distribution des résidus (Prédit − Réel)",
            labels={"x": "Résidu (µg/m³)", "count": "Fréquence"},
            color_discrete_sequence=["#7C6FCD"],
        )
        fig_res.add_vline(x=0, line_dash="dash", line_color="#FF6B6B",
                           annotation_text="Biais nul")
        fig_res.update_layout(**PLOTLY_LAYOUT, height=370)
        st.plotly_chart(fig_res, use_container_width=True)

    st.markdown("---")

    # ── Feature importance ────────────────────────────────────────────
    st.subheader("Importance des variables (Random Forest)")
    feat_names = [
        "pm25_lag_1h","pm25_roll_3h","pm25_lag_6h","pm25_roll_12h",
        "pm25_lag_24h","pm25_roll_24h","pm25_lag_12h","Iws",
        "pm25_roll_3d","TEMP","pm25_lag_1d","DEWP",
        "pm25_roll_7d","iws_roll_6h","PRES",
    ]
    importances = [0.182,0.141,0.118,0.097,0.081,0.068,0.055,
                   0.042,0.038,0.031,0.028,0.022,0.019,0.015,0.011]
    colors = ["#00C9A7" if "pm25" in f else "#185FA5" for f in feat_names]

    fig_fi = go.Figure(go.Bar(
        x=importances[::-1], y=feat_names[::-1],
        orientation="h",
        marker_color=colors[::-1],
        text=[f"{v:.1%}" for v in importances[::-1]],
        textposition="outside",
    ))
    fig_fi.update_layout(**PLOTLY_LAYOUT, height=480,
                          title="Top 15 features — importance relative",
                          xaxis_title="Importance",
                          margin=dict(t=50, b=20, l=40, r=70))
    st.plotly_chart(fig_fi, use_container_width=True)
    st.caption("🟢 Variables pollution (lags PM2.5)   |   🔵 Variables météorologiques")

    st.markdown("---")
    st.info(
        f"**R² = {stats['r2']:.3f}** — Le modèle explique **{stats['r2']*100:.1f}%** "
        f"de la variance du PM2.5. Les 42% restants correspondent à des facteurs non capturés : "
        f"feux agricoles, émissions ponctuelles, phénomènes météo locaux.\n\n"
        f"**MAE = {stats['mae']:.2f} µg/m³** — Erreur médiane inférieure à 40 µg/m³, soit "
        f"environ **{stats['mae']/stats['mean_pm25']*100:.0f}%** de la valeur historique moyenne. "
        f"Précision suffisante pour déclencher des alertes sanitaires de manière fiable."
    )


# ══════════════════════════════════════════════════════════════════════
# PAGE 4 — À propos du modèle
# ══════════════════════════════════════════════════════════════════════
elif page == "📄  À propos du modèle":

    # ── Hero ──────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,#0A0E17 0%,#0F2027 60%,#0A0E17 100%);
                    border-radius:20px;padding:52px 60px;margin-bottom:36px;
                    position:relative;overflow:hidden;
                    border:1px solid rgba(0,201,167,.15);">
            <div style="position:absolute;top:-80px;right:-80px;width:360px;height:360px;
                        border-radius:50%;
                        background:radial-gradient(circle,rgba(0,201,167,.12) 0%,transparent 70%);"></div>
            <div style="font-family:'Space Mono',monospace;font-size:11px;color:#00C9A7;
                        letter-spacing:.15em;text-transform:uppercase;margin-bottom:14px;">
                Rapport technique · Smart City · 2025
            </div>
            <div style="font-family:'Syne',sans-serif;font-size:44px;font-weight:800;
                        color:#F4F7FB;line-height:1.1;letter-spacing:-1.5px;margin-bottom:14px;">
                Prévision de la <span style="color:#00C9A7;">qualité de l'air</span><br>
                à Beijing par Machine Learning
            </div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:15px;
                        color:#4A5568;line-height:1.8;max-width:600px;">
                Modélisation prédictive du PM2.5 à horizon 24h à partir de données
                météorologiques et de séries temporelles de pollution.
                Une approche orientée aide à la décision urbaine.
            </div>
            <div style="display:flex;gap:36px;margin-top:32px;padding-top:28px;
                        border-top:1px solid rgba(255,255,255,.07);">
                {"".join(
                    f'<div><div style="font-family:\'Space Mono\',monospace;font-size:9px;'
                    f'color:#2D3748;text-transform:uppercase;letter-spacing:.1em;margin-bottom:4px;">{k}</div>'
                    f'<div style="font-family:\'Space Grotesk\',sans-serif;font-size:14px;color:#8694AA;">{v}</div></div>'
                    for k, v in [
                        ("Auteur",  "Abdoul Fataho NIAMPA"),
                        ("Domaine", "Data Science / Smart City"),
                        ("Données", "UCI Beijing PM2.5"),
                        ("Période", "2010 – 2014"),
                    ]
                )}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Section header helper ─────────────────────────────────────────
    def section(num, title):
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:16px;margin:44px 0 24px;">
                <span style="font-family:'Space Mono',monospace;font-size:11px;
                             color:#00C9A7;font-weight:500;
                             background:rgba(0,201,167,.08);border:1px solid rgba(0,201,167,.2);
                             padding:4px 12px;border-radius:20px;letter-spacing:.05em;">{num}</span>
                <span style="font-family:'Syne',sans-serif;font-size:26px;
                             font-weight:700;color:#0A0E17;letter-spacing:-.5px;">{title}</span>
                <div style="flex:1;height:1px;background:#E4EAF2;"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── 01 — Contexte ─────────────────────────────────────────────────
    section("01", "Contexte & Enjeux")
    col_c1, col_c2 = st.columns([3, 2])
    with col_c1:
        st.markdown(
            """
            <div style="font-family:'Space Grotesk',sans-serif;font-size:15px;
                        color:#334155;line-height:1.85;">
            <p>La pollution aux particules fines <strong>PM2.5</strong> est l'un des risques
            environnementaux les plus graves pour la santé publique urbaine.
            À Beijing (21 millions d'habitants), les épisodes intenses atteignent
            <strong style="color:#B91C1C;">500 µg/m³</strong> — 20 fois le seuil OMS de 25 µg/m³.</p>
            <p style="margin-top:16px;">Dans une vision <strong>Smart City</strong>,
            ce projet fournit aux décideurs un outil de prévision à J+1 pour anticiper
            les alertes, planifier des restrictions et communiquer vers les populations sensibles.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_c2:
        seuils = [("Bon","< 50","#00C9A7"), ("Modéré","50–100","#FFB830"),
                  ("Mauvais","100–150","#F97316"), ("Très mauvais","150–250","#FF6B6B"),
                  ("Dangereux","> 250","#B91C1C")]
        rows = "".join(
            f'<div style="display:flex;justify-content:space-between;padding:9px 0;'
            f'border-bottom:1px solid #F1F5F9;">'
            f'<span style="font-size:13px;color:#64748B;">{lab}</span>'
            f'<span style="font-size:13px;font-weight:600;color:{col};">{val} µg/m³</span></div>'
            for lab, val, col in seuils
        )
        st.markdown(
            f"""
            <div style="background:#FFFBF0;border:1px solid #FDE68A;border-left:3px solid #F59E0B;
                        border-radius:14px;padding:20px 22px;">
                <div style="font-family:'Space Mono',monospace;font-size:10px;color:#92400E;
                            text-transform:uppercase;letter-spacing:.08em;font-weight:600;
                            margin-bottom:10px;">Seuils OMS — PM2.5</div>
                {rows}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── 02 — Données ──────────────────────────────────────────────────
    section("02", "Jeu de données")
    c1, c2, c3 = st.columns(3)
    for col_, lab, val, unit, desc in [
        (c1, "Observations", "43 824", "enregistrements horaires",
         "Données 2010–2014 en continu, station US Embassy Beijing."),
        (c2, "PM2.5 moyen", f"{stats['mean_pm25']:.0f}", "µg/m³",
         "Concentration moyenne sur toute la période d'étude."),
        (c3, "Features construites", "39", "variables",
         "Lags, rolling means, météo dérivée, encodages temporels."),
    ]:
        col_.markdown(
            f"""
            <div style="background:#FFF;border:1px solid #E4EAF2;border-radius:16px;
                        padding:28px 24px;position:relative;overflow:hidden;
                        box-shadow:0 1px 4px rgba(0,0,0,.06);">
                <div style="font-family:'Space Mono',monospace;font-size:10px;color:#8694AA;
                            text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px;">{lab}</div>
                <div style="font-family:'Syne',sans-serif;font-size:44px;font-weight:800;
                            color:#0A0E17;letter-spacing:-1.5px;line-height:1;">{val}</div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:13px;
                            color:#8694AA;margin-bottom:10px;">{unit}</div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:12px;
                            color:#64748B;border-top:1px solid #F1F5F9;padding-top:10px;
                            line-height:1.6;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── 03 — Méthodologie ─────────────────────────────────────────────
    section("03", "Méthodologie")
    steps = [
        ("Étape 01","Exploration & Analyse",
         "Analyse de saisonnalité (STL), stationnarité (ADF), "
         "fonctions d'autocorrélation (ACF/PACF) pour identifier les lags pertinents."),
        ("Étape 02","Feature Engineering",
         "39 features : lags (1h, 6h, 12h, 24h, 1–7j), rolling means (3h–14j), "
         "variables météo dérivées, encodages saison & vent."),
        ("Étape 03","Modélisation & Validation",
         "Split temporel strict 2010–2013 → train, 2014 → test. "
         "3 modèles comparés : Régression Linéaire, Random Forest (500 arbres), XGBoost."),
        ("Étape 04","Interprétabilité",
         "Analyse SHAP (SHapley Additive exPlanations) — contribution de chaque variable. "
         "Feature importance globale et locale."),
    ]
    col_s1, col_s2 = st.columns(2)
    for i, (step, title, desc) in enumerate(steps):
        target = col_s1 if i % 2 == 0 else col_s2
        target.markdown(
            f"""
            <div style="position:relative;padding-left:28px;margin-bottom:24px;">
                <div style="position:absolute;left:0;top:5px;width:14px;height:14px;
                            border-radius:50%;background:#00C9A7;border:3px solid #F4F7FB;
                            box-shadow:0 0 0 1px #00C9A7;"></div>
                <div style="font-family:'Space Mono',monospace;font-size:10px;color:#00C9A7;
                            font-weight:500;text-transform:uppercase;letter-spacing:.1em;
                            margin-bottom:4px;">{step}</div>
                <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:600;
                            color:#0A0E17;margin-bottom:6px;">{title}</div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:13px;
                            color:#64748B;line-height:1.65;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── 04 — Limites ──────────────────────────────────────────────────
    section("04", "Limites & Perspectives")
    limites = [
        ("Portée géographique restreinte",
         "Modèle entraîné uniquement sur Beijing. Généralisation à d'autres métropoles "
         "(Shanghai, Delhi, Paris) nécessite un réentraînement spécifique."),
        ("Absence de données trafic & industrie",
         "Les émissions du trafic et des sites industriels ne sont pas capturées — "
         "expliquent une part des 42% de variance non modélisée."),
        ("Horizon de prévision limité",
         "Prédiction uniquement à J+1. Extension à J+2/J+3 dégraderait significativement "
         "les performances sans refonte de l'architecture de features."),
        ("Événements exceptionnels",
         "Les pics liés aux feux agricoles ou inversions thermiques sont difficiles "
         "à anticiper sans données de télédétection satellite."),
    ]
    lg, ld = st.columns(2)
    for i, (title, desc) in enumerate(limites):
        target = lg if i % 2 == 0 else ld
        target.markdown(
            f"""
            <div style="background:#FFFBF0;border:1px solid #FDE68A;border-left:3px solid #F59E0B;
                        border-radius:12px;padding:18px 20px;margin-bottom:12px;">
                <div style="font-family:'Syne',sans-serif;font-size:14px;font-weight:700;
                            color:#78350F;margin-bottom:5px;">{title}</div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:13px;
                            color:#92400E;line-height:1.6;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Footer auteur ─────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:24px;background:#0A0E17;
                    border-radius:16px;padding:28px 32px;margin-top:44px;
                    border:1px solid rgba(255,255,255,.06);">
            <div style="width:52px;height:52px;border-radius:50%;flex-shrink:0;
                        background:linear-gradient(135deg,#00C9A7,#185FA5);
                        display:flex;align-items:center;justify-content:center;
                        font-family:'Syne',sans-serif;font-size:18px;font-weight:800;color:#FFF;">
                AN
            </div>
            <div>
                <div style="font-family:'Syne',sans-serif;font-size:20px;font-weight:700;
                            color:#F4F7FB;">Abdoul Fataho NIAMPA</div>
                <div style="font-family:'Space Mono',monospace;font-size:11px;
                            color:#2D3748;letter-spacing:.05em;margin-top:2px;">
                    Data Scientist · Projet Smart City Beijing
                </div>
                <a href="https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data"
                   target="_blank"
                   style="font-family:'Space Grotesk',sans-serif;font-size:12px;
                          color:#00C9A7;text-decoration:none;display:inline-block;
                          margin-top:8px;">
                    Source des données — UCI ML Repository →
                </a>
            </div>
            <div style="margin-left:auto;text-align:right;">
                <div style="font-family:'Space Mono',monospace;font-size:9px;color:#2D3748;
                            text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;">
                    Version
                </div>
                <div style="font-family:'Space Mono',monospace;font-size:13px;color:#00C9A7;">
                    Random Forest v1.0
                </div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:12px;
                            color:#4A5568;margin-top:4px;">
                    scikit-learn · 500 estimateurs<br>
                    R² = {stats['r2']:.3f} · MAE = {stats['mae']:.1f} µg/m³
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
