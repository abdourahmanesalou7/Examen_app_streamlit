import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(page_title="Dashboard Salaires Maroc", layout="wide")

# Style CSS professionnel
st.markdown("""
<style>
    .block-container {
        padding: 2rem 3rem;
    }
    .title {
        font-size: 3rem;
        font-weight: 800;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #34495e;
        font-weight: 600;
    }
    .stButton>button {
        background-color: #2980b9;
        color: white;
        font-weight: bold;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #3498db;
    }
    .stDownloadButton>button {
        background-color: #27ae60;
        color: white;
        font-weight: bold;
        border-radius: 5px;
    }
    .stDownloadButton>button:hover {
        background-color: #2ecc71;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">\U0001F4CA Dashboard interactif : Analyse des salaires au Maroc</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar : chargement et filtres
with st.sidebar:
    st.header("\U0001F4C1 Charger un fichier CSV.")
    uploaded_file = st.file_uploader("", type="csv")

if uploaded_file is None:
    st.info("\U0001F4CC Veuillez importer un fichier CSV pour commencer l'analyse.")
    st.stop()

try:
    data = pd.read_csv(uploaded_file)
    with st.sidebar:
        st.success("\u2705 Fichier charg\u00e9 avec succ\u00e8s")
except Exception as e:
    st.error(f"\u274C Erreur lors de la lecture du fichier : {e}")
    st.stop()

# Filtres
with st.sidebar:
    st.subheader("\U0001F50D Filtres dynamiques")
    col_filters = data.select_dtypes(include='object').columns.tolist()
    filters = {}
    for col in col_filters:
        uniques = data[col].dropna().unique().tolist()
        if 1 < len(uniques) < 30:
            selected = st.multiselect(f"{col} ({len(uniques)} options)", options=uniques, default=uniques)
            filters[col] = selected

# Application des filtres
filtered_data = data.copy()
for col, values in filters.items():
    filtered_data = filtered_data[filtered_data[col].isin(values)]

# Bouton de téléchargement
csv = filtered_data.to_csv(index=False).encode('utf-8')
st.sidebar.download_button("\U0001F4E5 T\u00e9l\u00e9charger les donn\u00e9es filtr\u00e9es", data=csv, file_name='donnees_filtrees.csv', mime='text/csv')

# Définir les onglets
list_col = filtered_data.columns.tolist()
numeric_cols = filtered_data.select_dtypes(include=np.number).columns.tolist()
tabs = st.tabs(["\U0001F4CC Aper\u00e7u", "\U0001F9E0 Analyse colonne", "\U0001F501 Analyse croisée", "\U0001F4C8 Corrélation", "\U0001F30D Carte des villes", "\u2728 Prédiction"])

# Tab 1 : Aperçu
with tabs[0]:
    st.markdown('<h2 class="subtitle">\U0001F4CB Aper\u00e7u des donn\u00e9es filtr\u00e9es</h2>', unsafe_allow_html=True)
    st.dataframe(filtered_data, use_container_width=True)
    st.markdown('<h2 class="subtitle">\U0001F4CA Statistiques descriptives</h2>', unsafe_allow_html=True)
    st.dataframe(filtered_data.describe(include='all'), use_container_width=True)

# Tab 2 : Analyse colonne
with tabs[1]:
    st.markdown('<h2 class="subtitle">\U0001F9E0 Analyse d\'une variable</h2>', unsafe_allow_html=True)
    selected_col = st.selectbox("S\u00e9lectionnez une colonne :", list_col)
    if selected_col:
        st.write(filtered_data[selected_col].describe())
        if pd.api.types.is_numeric_dtype(filtered_data[selected_col]):
            fig = px.histogram(filtered_data, x=selected_col, nbins=30, title=f"Distribution de {selected_col}", color_discrete_sequence=['#00b894'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            counts = filtered_data[selected_col].value_counts().head(10).reset_index()
            counts.columns = [selected_col, 'Total']
            fig = px.bar(counts, x=selected_col, y='Total', title=f"Top 10 des valeurs de {selected_col}", color='Total')
            st.plotly_chart(fig, use_container_width=True)

# Tab 3 : Analyse croisée
with tabs[2]:
    st.markdown('<h2 class="subtitle">\U0001F501 Analyse croisée</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        col_x = st.selectbox("Variable X", list_col, key='x')
    with col2:
        col_y = st.selectbox("Variable Y", list_col, key='y')
    if col_x and col_y and col_x != col_y:
        if pd.api.types.is_numeric_dtype(filtered_data[col_x]) and pd.api.types.is_numeric_dtype(filtered_data[col_y]):
            fig = px.scatter(filtered_data, x=col_x, y=col_y, trendline="ols", title=f"{col_x} vs {col_y}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Les deux colonnes doivent \u00eatre num\u00e9riques.")

# Tab 4 : Corrélation
with tabs[3]:
    st.markdown('<h2 class="subtitle">📈 Corrélation avancée</h2>', unsafe_allow_html=True)

    if len(numeric_cols) >= 2:
        selected_corr_cols = st.multiselect("Variables à inclure :", numeric_cols, default=numeric_cols)
        threshold = st.slider("Seuil de corrélation minimale (absolue)", 0.0, 1.0, 0.7, 0.05)

        corr_data = filtered_data[selected_corr_cols].dropna()
        corr_matrix = corr_data.corr()

        # Filtrer la matrice selon le seuil
        mask = corr_matrix.abs() >= threshold
        filtered_corr = corr_matrix[mask]

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(filtered_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    linewidths=0.5, square=True, cbar_kws={"shrink": 0.75})
        ax.set_title("Matrice de corrélation filtrée", fontsize=16)
        st.pyplot(fig)

        if 'Salaire' in filtered_data.columns:
            st.markdown("### 🧠 Top 3 variables les plus corrélées avec le Salaire")
            salaire_corr = corr_matrix['Salaire'].drop('Salaire').abs().sort_values(ascending=False)
            top_vars = salaire_corr.head(3).index.tolist()
            for var in top_vars:
                fig = px.scatter(filtered_data, x=var, y='Salaire', trendline="ols", title=f"{var} vs Salaire")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("❗ Pas assez de colonnes numériques pour calculer une corrélation.")

# Tab 5 : Carte des villes
with tabs[4]:
    st.markdown('<h2 class="subtitle">\U0001F30D Carte des salaires moyens par ville</h2>', unsafe_allow_html=True)
    location_cols = [col for col in data.columns if 'ville' in col.lower() or 'localisation' in col.lower()]
    if location_cols and 'Salaire' in data.columns:
        ville_col = location_cols[0]
        map_data = filtered_data.groupby(ville_col)['Salaire'].mean().reset_index()
        fig = px.bar(map_data, x=ville_col, y='Salaire', color='Salaire', title="Salaire moyen par ville")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("\u274C Donn\u00e9es de localisation ou Salaire manquantes.")

# Tab 6 : Prédiction
with tabs[5]:
    st.markdown('<h2 class="subtitle">\u2728 Estimation du salaire</h2>', unsafe_allow_html=True)
    if 'Salaire' not in data.columns:
        st.warning("Colonne 'Salaire' requise pour la pr\u00e9diction.")
    else:
        df_clean = data.dropna(subset=['Salaire'])
        X = df_clean.drop(columns=['Salaire'])
        y = df_clean['Salaire']

        cat_cols = X.select_dtypes(include='object').columns.tolist()
        num_cols = X.select_dtypes(include=np.number).columns.tolist()

        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ], remainder='passthrough')

        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        model.fit(X, y)

        st.markdown("#### \U0001F4DD Entrez vos crit\u00e8res :")
        user_input = {}
        with st.form(key='form_pred'):
            col1, col2 = st.columns(2)
            with col1:
                for col in cat_cols[:len(cat_cols)//2 + 1]:
                    user_input[col] = st.selectbox(f"{col} :", sorted(X[col].dropna().unique()))
            with col2:
                for col in cat_cols[len(cat_cols)//2 + 1:]:
                    user_input[col] = st.selectbox(f"{col} :", sorted(X[col].dropna().unique()))
                for col in num_cols:
                    val_min, val_max = int(X[col].min()), int(X[col].max())
                    user_input[col] = st.slider(f"{col} :", val_min, val_max, int((val_min + val_max) // 2))
            submit_button = st.form_submit_button("\U0001F50D Pr\u00e9dire")

        if submit_button:
            try:
                input_df = pd.DataFrame([user_input])
                prediction = model.predict(input_df)[0]
                st.success(f"\U0001F4B0 Salaire estim\u00e9 : {round(prediction, 2)} MAD")
            except Exception as e:
                st.error(f"Erreur : {e}")

        st.markdown("### \U0001F4C9 R\u00e9el vs Pr\u00e9dit")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        compare_df = pd.DataFrame({"Réel": y_test, "Prédit": y_pred})
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=compare_df['Réel'], mode='lines+markers', name='Réel'))
        fig.add_trace(go.Scatter(y=compare_df['Prédit'], mode='lines+markers', name='Prédit'))
        fig.update_layout(title="Comparaison Réel vs Prédit", xaxis_title="Échantillons", yaxis_title="Salaire (MAD)")
        st.plotly_chart(fig, use_container_width=True)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.metric("Erreur absolue moyenne (MAE)", f"{mae:.2f} MAD")
        st.metric("Score R²", f"{r2:.2f}")