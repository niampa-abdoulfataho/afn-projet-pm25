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
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/"
    "thumb/f/fa/Beijing_montage.jpg/320px-Beijing_montage.jpg",
    use_column_width=True
)
st.sidebar.title("🏙️ Smart City Beijing")
st.sidebar.markdown("**Prévision PM2.5 — J+1**")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Accueil & Prédiction",
     "📊 Historique & Tendances",
     "🔍 Analyse des performances",
     "ℹ️ À propos du modèle"]
)

# ══════════════════════════════════════════════════════════════
# PAGE 1 — Accueil & Prédiction
# ══════════════════════════════════════════════════════════════
if page == "🏠 Accueil & Prédiction":

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
        "pm25_lag_12h"    : pm25_roll_12h,
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
elif page == "📊 Historique & Tendances":

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
elif page == "🔍 Analyse des performances":

    st.title("🔍 Performances du modèle sur 2014")

    # Recalcul prédictions sur test set
    test_df = df[df.index.year == 2014].copy()

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
# PAGE 4 — À propos
# ══════════════════════════════════════════════════════════════
elif page == "ℹ️ À propos du modèle":

    st.title("ℹ️ À propos du projet")

    st.markdown("""
    ## 🏙️ Prévision PM2.5 — Smart City Beijing

    ### Contexte
    Ce projet s'inscrit dans le cadre d'une **Smart City** : utiliser
    les données et le Machine Learning pour anticiper les épisodes de
    pollution et aider les décideurs urbains à prendre des mesures
    proactives.

    ### Données
    - **Source** : UCI Beijing PM2.5 Dataset
    - **Période** : 2010–2014 (43 824 observations horaires)
    - **Variables** : météo (température, vent, pression, humidité)
      + pollution (PM2.5)

    ### Méthodologie
    1. **Exploration** : analyse de saisonnalité, ACF/PACF,
       décomposition STL
    2. **Feature Engineering** : 32 features (lags, rolling means,
       variables météo dérivées, encodages)
    3. **Modélisation** : Régression Linéaire (baseline),
       Random Forest, XGBoost — split temporel 2010–2013 / 2014
    4. **Interprétabilité** : SHAP values, Feature Importance

    ### Modèle retenu
    **Random Forest** — R²=0.568, RMSE=52.12 µg/m³, MAE=37.26 µg/m³

    ### Limites
    - Données limitées à Beijing
    - Absence de données trafic et émissions industrielles
    - Horizon de prévision : 24h uniquement

    ### Auteur
    **Abdoul Fataho NIAMPA** — Projet Data Scientist Smart City
    """)

    st.markdown("---")
    st.markdown(
        "📂 Dataset : "
        "[UCI ML Repository](https://archive.ics.uci.edu/ml/"
        "datasets/Beijing+PM2.5+Data)"
    )






