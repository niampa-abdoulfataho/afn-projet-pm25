# 🏙️ Smart City Beijing — Prévision PM2.5

> **Application Streamlit de prévision du niveau de particules fines PM2.5 à Beijing à horizon J+1**
> Projet fil conducteur — Data Science / Smart City
> Auteur : **Abdoul Fataho NIAMPA**

---

## 📌 Description

Cette application prédit la concentration en **PM2.5** (particules fines < 2,5 µm)
à Beijing pour le lendemain (J+1) à partir des conditions météorologiques et de
pollution actuelles.

Elle s'appuie sur un modèle **Random Forest** (scikit-learn) entraîné sur le
dataset historique [UCI Beijing PM2.5](https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data)
couvrant **43 824 observations horaires** de 2010 à 2014.

---

## 🗂️ Structure du projet

```
afn-projet-pm25/
│
├── app.py                    # Application Streamlit principale
├── model_pm25.pkl            # Modèle Random Forest sérialisé (joblib)
├── features.json             # Liste ordonnée des 39 features du modèle
├── model_stats.json          # Métriques de performance (RMSE, MAE, R², moyenne PM2.5)
├── beijing_features.csv      # Dataset historique enrichi (optionnel — données synthétiques si absent)
│
├── Projet_Fil_Conducteur_Abdoul_Fataho_NIAMPA.ipynb   # Notebook d'entraînement complet
├── requirements.txt          # Dépendances Python
└── README.md                 # Ce fichier
```

---

## 🚀 Installation & Lancement

### 1. Cloner le dépôt

```bash
git clone https://github.com/<votre-username>/afn-projet-pm25.git
cd afn-projet-pm25
```

### 2. Créer un environnement virtuel (recommandé)

```bash
python -m venv venv
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvre automatiquement sur `http://localhost:8501`.

---

## 📱 Pages de l'application

| Page | Description |
|------|-------------|
| 🏠 **Accueil & Prédiction** | Formulaire de saisie des conditions actuelles → prédiction PM2.5 J+1 avec jauge et conseil opérationnel |
| 📊 **Historique & Tendances** | Exploration EDA : série temporelle, saisonnalité, heatmap, corrélations météo |
| 🎯 **Performances du modèle** | Métriques, tableau comparatif 3 modèles, prédit vs réel, feature importance |
| 📄 **À propos du modèle** | Rapport complet : contexte, méthodologie, limites, perspectives |

---

## 🧠 Modèle & Features

### Algorithme retenu

**Random Forest Regressor** — scikit-learn
- 500 estimateurs
- Split temporel strict : train 2010–2013 / test 2014

### Métriques de performance (jeu test 2014)

| Métrique | Valeur |
|----------|--------|
| RMSE | 51.76 µg/m³ |
| MAE  | 37.16 µg/m³ |
| R²   | 0.574 |

### Features (39 variables)

| Catégorie | Variables |
|-----------|-----------|
| Lags PM2.5 | `pm25_lag_1h`, `pm25_lag_6h`, `pm25_lag_12h`, `pm25_lag_24h`, `pm25_lag_1d/2d/3d/7d` |
| Rolling means | `pm25_roll_3h`, `pm25_roll_12h`, `pm25_roll_24h`, `pm25_roll_3d/7d/14d` |
| Météo brute | `TEMP`, `PRES`, `DEWP`, `Iws`, `Is`, `Ir` |
| Météo dérivée | `temp_feels_like`, `delta_temp`, `temp_roll_6h`, `iws_roll_6h`, `temp_x_vent`, `pres_x_dewp` |
| Temporel | `month`, `dayofweek`, `is_weekend`, `saison_*` |
| Vent encodé | `wind_dir_NW`, `wind_dir_SE`, `wind_dir_cv`, `vent_modéré`, `vent_fort` |
| Delta | `pm25_delta_1d`, `is_rainy` |

---

## 📊 Dataset source

- **Nom** : Beijing PM2.5 Data Set
- **Source** : [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data)
- **Période** : 1er janvier 2010 — 31 décembre 2014
- **Fréquence** : Horaire
- **Observations** : 43 824

---

## ⚠️ Limites connues

- Modèle entraîné exclusivement sur **Beijing** — non généralisable directement.
- Absence de données **trafic** et **émissions industrielles**.
- Horizon de prévision limité à **J+1** (24h).
- Pics extrêmes liés aux **feux agricoles** difficiles à anticiper.

---

## 🛠️ Stack technique

- **Python** 3.10+
- **Streamlit** 1.32 — interface web
- **scikit-learn** 1.3 — modèle Random Forest
- **Plotly** 5.18 — visualisations interactives
- **Pandas / NumPy** — manipulation de données
- **joblib** — sérialisation du modèle

---

## 📄 Licence

Projet académique — usage libre pour la recherche et l'enseignement.
Données source : UCI ML Repository (domaine public).
