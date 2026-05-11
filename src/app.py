import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="We vs Them Analysis", page_icon="🛡️", layout="wide")

# Modern Premium CSS
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .main-title {
        font-size: 3.2rem; font-weight: 800;
        background: linear-gradient(90deg, #ff4b4b, #ff8f00);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0px; padding-bottom: 0px;
    }
    .subtitle { color: #8b949e; font-size: 1.2rem; margin-bottom: 30px; }
    [data-testid="stMetricValue"] { font-size: 2.5rem !important; font-weight: 800 !important; color: #ffffff !important; }
    [data-testid="stMetricLabel"] { font-size: 1.1rem !important; color: #8b949e !important; text-transform: uppercase; font-weight: 600; }
    [data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #30363d; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; font-size: 1.2rem; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { background-color: rgba(255, 75, 75, 0.1) !important; color: #ff4b4b !important; border-bottom: 2px solid #ff4b4b !important;}
    </style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=60)
def load_data():
    dfs = []
    processed_dir = "data/processed"
    scored_dir = "data/scored"
    
    for root, _, files in os.walk(processed_dir):
        for file in files:
            if file.endswith(".csv"):
                rel_path = os.path.relpath(os.path.join(root, file), processed_dir)
                scored_file = os.path.join(scored_dir, rel_path)
                
                try:
                    if os.path.exists(scored_file):
                        df = pd.read_csv(scored_file, low_memory=False)
                        df['is_scored'] = True
                    else:
                        df = pd.read_csv(os.path.join(root, file), low_memory=False)
                        df['toxicity'] = 0.0
                        df['sentiment'] = 'neutral'
                        df['is_scored'] = False
                        
                    if 'clean_text' in df.columns and 'is_polarized' in df.columns:
                        path_parts = os.path.normpath(rel_path).split(os.sep)
                        platform = path_parts[0].capitalize() if len(path_parts) > 1 else "General"
                        df['platform'] = platform
                        dfs.append(df)
                except Exception as e:
                    pass
                    
    if not dfs:
        return pd.DataFrame()
    
    final_df = pd.concat(dfs, ignore_index=True)
    # Remove any duplicate comments that might exist across different CSV files
    final_df = final_df.drop_duplicates(subset=['clean_text'])
    
    return final_df

df = load_data()

st.markdown("<h1 class='main-title'>🛡️ Project Shield Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Interactive monitoring of sports and social commentary polarization.</p>", unsafe_allow_html=True)

if df.empty:
    st.error("No data found. Please run linguistic_analysis.py first.")
    st.stop()

# --- SIDEBAR FILTERS ---
st.sidebar.title("🎛️ Filters")
platforms = ["All"] + sorted(list(df['platform'].unique()))
selected_platform = st.sidebar.selectbox("Select Platform", platforms)

if selected_platform != "All":
    filtered_df = df[df['platform'] == selected_platform]
else:
    filtered_df = df

st.sidebar.markdown("---")
scored_percent = (filtered_df['is_scored'].sum() / len(filtered_df)) * 100
if scored_percent < 100:
    st.sidebar.warning(f"⏳ AI Scoring in progress... ({scored_percent:.1f}%). Refresh for updates.")
else:
    st.sidebar.success("✅ AI Scoring Complete")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Vue Globale", "🚨 Analyse Toxicité", "🔍 Explorateur de Données"])

with tab1:
    st.header("Vue Globale des Données")
    
    # KPIs
    col1, col2, col3 = st.columns(3)
    pol_pct = filtered_df['is_polarized'].mean() * 100
    with col1: st.metric("Messages Analysés", f"{len(filtered_df):,}")
    with col2: st.metric("Taux de Polarisation", f"{pol_pct:.1f}%")
    with col3: st.metric("Score de Toxicité Moyen", f"{filtered_df['toxicity'].mean():.3f}")
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Répartition par Plateforme")
        if selected_platform == "All":
            plat_counts = df['platform'].value_counts().reset_index()
            plat_counts.columns = ['Plateforme', 'Nombre']
            fig_plat = px.pie(plat_counts, values='Nombre', names='Plateforme', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
            fig_plat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9")
            st.plotly_chart(fig_plat, use_container_width=True)
        else:
            st.info("Sélectionnez 'All' pour voir la répartition globale.")

    with c2:
        st.subheader("Distribution des Sentiments")
        sent_counts = filtered_df[filtered_df['is_scored'] == True]['sentiment'].value_counts().reset_index()
        if not sent_counts.empty:
            sent_counts.columns = ['Sentiment', 'Nombre']
            color_map = {'positive': '#2e7d32', 'neutral': '#1f77b4', 'negative': '#d32f2f'}
            fig_sent = px.bar(sent_counts, x='Sentiment', y='Nombre', color='Sentiment', color_discrete_map=color_map)
            fig_sent.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9", showlegend=False)
            st.plotly_chart(fig_sent, use_container_width=True)
        else:
            st.info("Attente des résultats de l'IA...")

with tab2:
    st.header("Analyse de l'Impact de la Polarisation")
    
    pol_df = filtered_df[filtered_df['is_polarized'] == True]
    non_pol_df = filtered_df[filtered_df['is_polarized'] == False]
    
    avg_tox_pol = pol_df['toxicity'].mean() if not pol_df.empty else 0
    avg_tox_non_pol = non_pol_df['toxicity'].mean() if not non_pol_df.empty else 0
    
    if scored_percent > 0:
        fig_tox = px.bar(
            x=["Normal", "Polarisé ('Nous vs Eux')"], 
            y=[avg_tox_non_pol, avg_tox_pol],
            color=["Normal", "Polarisé"],
            color_discrete_map={"Normal": "#1f77b4", "Polarisé": "#ff4b4b"},
            labels={'x': 'Type de Message', 'y': 'Score de Toxicité Moyen'},
            title="Comparaison Directe : L'impact du langage polarisé sur la toxicité"
        )
        fig_tox.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9", showlegend=False)
        st.plotly_chart(fig_tox, use_container_width=True)
    else:
        st.info("Données en cours de génération...")

    st.markdown("### 🚨 Top 10 Commentaires les plus Toxiques")
    top_toxic = filtered_df[filtered_df['is_scored'] == True].sort_values(by='toxicity', ascending=False).head(10)
    
    if top_toxic.empty:
        st.info("Aucun commentaire n'a encore été analysé.")
    else:
        for idx, row in top_toxic.iterrows():
            with st.expander(f"Toxicité: {row['toxicity']:.2f} | {row['platform']} | Sentiment: {row['sentiment']}"):
                st.write(f"**Texte original:** {row['clean_text']}")
                st.write(f"*Polarisé ?* {'Oui' if row['is_polarized'] else 'Non'}")

with tab3:
    st.header("Explorateur de Données Interactif")
    st.markdown("Filtrez et explorez vous-même le dataset complet pour trouver des exemples concrets pour votre rapport.")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    filter_pol = col_f1.selectbox("Type de message", ["Tous", "Polarisés uniquement", "Non Polarisés"])
    filter_sent = col_f2.selectbox("Sentiment", ["Tous", "negative", "neutral", "positive"])
    min_tox = col_f3.slider("Toxicité minimum", 0.0, 1.0, 0.0, 0.05)
    
    explore_df = filtered_df.copy()
    if filter_pol == "Polarisés uniquement":
        explore_df = explore_df[explore_df['is_polarized'] == True]
    elif filter_pol == "Non Polarisés":
        explore_df = explore_df[explore_df['is_polarized'] == False]
        
    if filter_sent != "Tous":
        explore_df = explore_df[explore_df['sentiment'] == filter_sent]
        
    explore_df = explore_df[explore_df['toxicity'] >= min_tox]
    
    display_df = explore_df[['platform', 'clean_text', 'is_polarized', 'sentiment', 'toxicity']].sort_values(by='toxicity', ascending=False)
    
    st.metric("Résultats trouvés", f"{len(display_df):,}")
    st.dataframe(display_df, use_container_width=True, height=500)
