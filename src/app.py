import os
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv
from rag.assistant import AIAssistant
from models.explainability import ModelExplainer
from rag.agent import FunctionCallingAgent

# Load environment variables
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token


class DataManager:
    """Handles loading, caching, and filtering the social dataset."""

    def __init__(self, processed_dir: str = "data/processed", scored_dir: str = "data/scored") -> None:
        self.processed_dir = processed_dir
        self.scored_dir = scored_dir

    def load_raw_or_scored(self, root: str, file: str) -> pd.DataFrame:
        """Loads scored file if it exists, otherwise falls back to processed file."""
        rel_path = os.path.relpath(os.path.join(root, file), self.processed_dir)
        scored_file = os.path.join(self.scored_dir, rel_path)

        if os.path.exists(scored_file):
            df = pd.read_csv(scored_file, low_memory=False)
            df['is_scored'] = True
            return df

        df = pd.read_csv(os.path.join(root, file), low_memory=False)
        df['toxicity'] = 0.0
        df['sentiment'] = 'neutral'
        df['is_scored'] = False
        return df

    def get_platform_name(self, rel_path: str) -> str:
        """Determines platform name based on directory structure."""
        path_parts = os.path.normpath(rel_path).split(os.sep)
        return path_parts[0].capitalize() if len(path_parts) > 1 else "General"

    def load_combined_data(self) -> pd.DataFrame:
        """Walks processed directory, merges files, and cleans duplicates."""
        if not os.path.exists(self.processed_dir):
            return pd.DataFrame()

        dfs = []
        for root, _, files in os.walk(self.processed_dir):
            for file in files:
                if not file.endswith(".csv"):
                    continue
                try:
                    df = self.load_raw_or_scored(root, file)
                    if 'clean_text' not in df.columns or 'is_polarized' not in df.columns:
                        continue

                    rel_path = os.path.relpath(os.path.join(root, file), self.processed_dir)
                    df['platform'] = self.get_platform_name(rel_path)
                    dfs.append(df)
                except Exception:
                    pass

        if not dfs:
            return pd.DataFrame()

        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df.drop_duplicates(subset=['clean_text'])

    def process_and_score_uploaded_csv(self, uploaded_file, platform_name: str) -> bool:
        """Processes and scores an uploaded CSV file, saving it directly to scored directory."""
        try:
            df = pd.read_csv(uploaded_file, low_memory=False)
            
            from preprocessing.linguistic_analysis import TextCleaner, PolarizationAnalyzer
            cleaner = TextCleaner()
            analyzer = PolarizationAnalyzer()
            
            possible_columns = [
                "childCommentText", "text", "Text",
                "video_transcription_text", "caption", "parentText"
            ]
            text_col = None
            for col in possible_columns:
                if col in df.columns:
                    text_col = col
                    break
            if not text_col:
                obj_cols = df.select_dtypes(include=['object']).columns
                text_col = obj_cols[0] if len(obj_cols) > 0 else df.columns[0]
            
            df['clean_text'] = df[text_col].fillna("").astype(str).apply(cleaner.clean)
            
            analysis_df = analyzer.analyze_batch(df['clean_text'].tolist())
            df = pd.concat([df.reset_index(drop=True), analysis_df.reset_index(drop=True)], axis=1)
            
            from models.sentiment_toxicity import ModelScorer
            scorer = ModelScorer()
            
            df['clean_text'] = df['clean_text'].str.strip()
            valid_mask = df['clean_text'] != ""
            valid_texts = df.loc[valid_mask, 'clean_text'].tolist()
            
            df['sentiment'] = 'neutral'
            df['toxicity'] = 0.0
            
            if valid_texts:
                sentiments, toxicities = scorer.score_texts(valid_texts)
                df.loc[valid_mask, 'sentiment'] = sentiments
                df.loc[valid_mask, 'toxicity'] = toxicities
            
            df['is_scored'] = True
            df['platform'] = platform_name.capitalize()
            
            platform_dir = os.path.join(self.scored_dir, platform_name.lower())
            os.makedirs(platform_dir, exist_ok=True)
            output_filepath = os.path.join(platform_dir, uploaded_file.name)
            df.to_csv(output_filepath, index=False)
            
            proc_platform_dir = os.path.join(self.processed_dir, platform_name.lower())
            os.makedirs(proc_platform_dir, exist_ok=True)
            df.to_csv(os.path.join(proc_platform_dir, uploaded_file.name), index=False)
            
            return True
        except Exception as e:
            print(f"Error processing uploaded CSV: {e}")
            return False



# Streamlit Setup
st.set_page_config(page_title="We vs Them Analysis", page_icon="🛡️", layout="wide")

# Modern Premium CSS Injection
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
def get_dashboard_data() -> pd.DataFrame:
    """Instantiates data manager and loads combined data."""
    manager = DataManager()
    return manager.load_combined_data()


df = get_dashboard_data()

st.markdown("<h1 class='main-title'>We vs Them Analysis Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Interactive monitoring of sports and social commentary polarization.</p>", unsafe_allow_html=True)

if df.empty:
    st.error("No data found. Please run linguistic_analysis.py first.")
    st.stop()

# --- SIDEBAR CONTROLS ---
st.sidebar.title("🎛️ Controls")

# Platform Filter
platforms = ["All"] + sorted(list(df['platform'].unique()))
selected_platform = st.sidebar.selectbox("Select Platform", platforms)

if selected_platform != "All":
    filtered_df = df[df['platform'] == selected_platform]
else:
    filtered_df = df

st.sidebar.markdown("---")

# CSV File Upload Section
st.sidebar.subheader("Upload New Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
upload_platform = st.sidebar.text_input("Source/Platform Name", "Custom")

if uploaded_file is not None:
    if st.sidebar.button("Process & Score Dataset"):
        with st.sidebar.status("Processing uploaded data...", expanded=True) as status:
            st.write("Extracting linguistic features & running AI sentiment/toxicity pipelines...")
            manager = DataManager()
            success = manager.process_and_score_uploaded_csv(uploaded_file, upload_platform)
            if success:
                st.write("Updating FAISS vector index...")
                from rag.vectorstore import VectorStoreManager
                v_manager = VectorStoreManager()
                v_manager.build_vector_store()
                
                status.update(label="✅ Success! Dataset Processed & Indexed", state="complete")
                st.sidebar.success("Data uploaded. Reloading...")
                st.cache_data.clear()
                st.rerun()
            else:
                status.update(label="❌ Processing failed", state="error")

st.sidebar.markdown("---")
scored_percent = (filtered_df['is_scored'].sum() / len(filtered_df)) * 100
if scored_percent < 100:
    st.sidebar.warning(f"⏳ AI Scoring in progress... ({scored_percent:.1f}%). Refresh for updates.")
else:
    st.sidebar.success("✅ AI Scoring Complete")

# --- TABS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Global Overview", 
    "Toxicity Analysis", 
    "Data Explorer", 
    "Fairness & Causal", 
    "XAI Explainer", 
    "AI Assistant"
])

with tab1:
    st.header("Global Data Overview")

    # KPIs
    col1, col2, col3 = st.columns(3)
    pol_pct = filtered_df['is_polarized'].mean() * 100
    with col1:
        st.metric("Analyzed Messages", f"{len(filtered_df):,}")
    with col2:
        st.metric("Polarization Rate", f"{pol_pct:.1f}%")
    with col3:
        st.metric("Average Toxicity Score", f"{filtered_df['toxicity'].mean():.3f}")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Distribution by Platform")
        if selected_platform == "All":
            plat_counts = df['platform'].value_counts().reset_index()
            plat_counts.columns = ['Platform', 'Count']
            fig_plat = px.pie(plat_counts, values='Count', names='Platform', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
            fig_plat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9")
            st.plotly_chart(fig_plat, use_container_width=True)
        else:
            st.info("Select 'All' to view global distribution.")

    with c2:
        st.subheader("Sentiment Distribution")
        sent_counts = filtered_df[filtered_df['is_scored'] == True]['sentiment'].value_counts().reset_index()
        if not sent_counts.empty:
            sent_counts.columns = ['Sentiment', 'Count']
            color_map = {'positive': '#2e7d32', 'neutral': '#1f77b4', 'negative': '#d32f2f'}
            fig_sent = px.bar(sent_counts, x='Sentiment', y='Count', color='Sentiment', color_discrete_map=color_map)
            fig_sent.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9", showlegend=False)
            st.plotly_chart(fig_sent, use_container_width=True)
        else:
            st.info("Waiting for AI results...")

with tab2:
    st.header("Polarization Impact Analysis")

    pol_df = filtered_df[filtered_df['is_polarized'] == True]
    non_pol_df = filtered_df[filtered_df['is_polarized'] == False]

    avg_tox_pol = pol_df['toxicity'].mean() if not pol_df.empty else 0
    avg_tox_non_pol = non_pol_df['toxicity'].mean() if not non_pol_df.empty else 0

    if scored_percent > 0:
        fig_tox = px.bar(
            x=["Normal", "Polarized ('Us vs Them')"],
            y=[avg_tox_non_pol, avg_tox_pol],
            color=["Normal", "Polarized"],
            color_discrete_map={"Normal": "#1f77b4", "Polarized": "#ff4b4b"},
            labels={'x': 'Message Type', 'y': 'Average Toxicity Score'},
            title="Direct Comparison: Impact of polarized language on toxicity"
        )
        fig_tox.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9", showlegend=False)
        st.plotly_chart(fig_tox, use_container_width=True)
    else:
        st.info("Generating data...")

    st.markdown("### Top 10 Most Toxic Comments")
    top_toxic = filtered_df[filtered_df['is_scored'] == True].sort_values(by='toxicity', ascending=False).head(10)

    if top_toxic.empty:
        st.info("No comments have been analyzed yet.")
    else:
        for idx, row in top_toxic.iterrows():
            with st.expander(f"Toxicity: {row['toxicity']:.4f} | {row['platform']} | Sentiment: {row['sentiment']}"):
                st.write(f"**Original text:** {row['clean_text']}")
                st.write(f"*Polarized?* {'Yes' if row['is_polarized'] else 'No'}")

with tab3:
    st.header("Interactive Data Explorer")
    st.markdown("Filter and explore the complete dataset yourself to find concrete examples for your report.")

    col_f1, col_f2, col_f3 = st.columns(3)
    filter_pol = col_f1.selectbox("Message Type", ["All", "Polarized only", "Non-polarized"])
    filter_sent = col_f2.selectbox("Sentiment", ["All", "negative", "neutral", "positive"])
    min_tox = col_f3.slider("Minimum Toxicity", 0.0, 1.0, 0.0, 0.05)

    explore_df = filtered_df.copy()
    if filter_pol == "Polarized only":
        explore_df = explore_df[explore_df['is_polarized'] == True]
    elif filter_pol == "Non-polarized":
        explore_df = explore_df[explore_df['is_polarized'] == False]

    if filter_sent != "All":
        explore_df = explore_df[explore_df['sentiment'] == filter_sent]

    explore_df = explore_df[explore_df['toxicity'] >= min_tox]

    display_df = explore_df[['platform', 'clean_text', 'is_polarized', 'sentiment', 'toxicity']].sort_values(by='toxicity', ascending=False)

    st.metric("Results Found", f"{len(display_df):,}")
    st.dataframe(display_df, use_container_width=True, height=500)

with tab4:
    st.header("Bias, Fairness & Causal Spikes")
    st.markdown("Auditing toxicity classification fairness metrics across platforms and tracing causal events.")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.subheader("Platform Bias & Fairness Report")
        report_path = "data/processed/fairness_report.txt"
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                st.text_area("Fairness Report Details", f.read(), height=400)
        else:
            st.warning("Fairness report not found. Run bias_check.py first.")
            
    with col_c2:
        st.subheader("Causal Event Graph")
        graph_path = "data/processed/causal_graph.png"
        if os.path.exists(graph_path):
            st.image(graph_path, caption="Causal Pathway Graph during Sports Events", use_container_width=True)
        else:
            st.warning("Causal graph not found. Run causal_analysis.py first.")
            
    st.markdown("---")
    st.subheader("Detected Temporal Activity Spikes")
    spikes_path = "data/processed/detected_spikes.csv"
    if os.path.exists(spikes_path):
        spikes_df = pd.read_csv(spikes_path)
        st.dataframe(spikes_df[['parsed_date', 'comment_count', 'event']], use_container_width=True, height=250)
    else:
        st.warning("No detected spikes dataset found.")

with tab5:
    st.header("Explainable AI (XAI) Explainer")
    st.markdown("Understand model decisions. This tab highlights token-level feature importances using LIME or SHAP (perturbation-based).")
    
    if "model_explainer" not in st.session_state:
        st.session_state.model_explainer = ModelExplainer()

    xai_input_mode = st.radio("Choose Input Method", ["Select Example from Data", "Input Custom Comment"])
    
    input_text = ""
    if xai_input_mode == "Select Example from Data":
        example_options = filtered_df[filtered_df['is_scored'] == True].sort_values(by='toxicity', ascending=False).head(20)['clean_text'].tolist()
        if example_options:
            input_text = st.selectbox("Select Comment to Explain", example_options)
        else:
            st.info("No scored comments available. Enter a custom comment.")
            xai_input_mode = "Input Custom Comment"
            
    if xai_input_mode == "Input Custom Comment":
        input_text = st.text_area("Enter sentence to explain:", "Oh, what a brilliant defense, letting them score in the last minute. /s")
        
    xai_method = st.selectbox("Explainability Method", ["LIME (Fast)", "SHAP (Deep Perturbation)"])
    
    if st.button("Generate Token Highlight Heatmap"):
        if not input_text.strip():
            st.warning("Please enter or select a comment.")
        else:
            with st.spinner("Calculating word feature importances..."):
                try:
                    st.session_state.model_explainer.initialize_pipeline()
                    
                    if xai_method == "LIME (Fast)":
                        weights = st.session_state.model_explainer.explain_lime(input_text)
                    else:
                        weights = st.session_state.model_explainer.explain_shap_perturbation(input_text)
                    
                    heatmap_html = st.session_state.model_explainer.generate_heatmap_html(input_text, weights)
                    
                    st.markdown("### Feature Importance Heatmap")
                    st.components.v1.html(heatmap_html, height=185, scrolling=False)
                    
                    st.markdown("### Semantically Similar Comments in FAISS Vector Store")
                    similar_docs = st.session_state.ai_assistant.retrieve_context_documents(input_text, k=4)
                    st.markdown(similar_docs)
                    
                except Exception as e:
                    st.error(f"XAI Error: {e}")

with tab6:
    st.header("AI Assistant")
    st.markdown("Ask the AI assistant about polarization, toxicity, or specific spikes.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "ai_assistant" not in st.session_state:
        st.session_state.ai_assistant = AIAssistant()

    if "react_agent" not in st.session_state:
        st.session_state.react_agent = FunctionCallingAgent()

    use_react_agent = st.checkbox("Activate ReAct Agent (runs Python analytical tools for complex queries)", value=False)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask something (e.g. 'Show me the top topics with highest polarization and also platform metrics for Twitter')"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing data and generating response..."):
                try:
                    if use_react_agent:
                        reply_text = st.session_state.react_agent.run(prompt)
                    else:
                        reply_text = st.session_state.ai_assistant.ask(
                            prompt,
                            st.session_state.messages,
                            filtered_df
                        )
                    st.markdown(reply_text)
                    st.session_state.messages.append({"role": "assistant", "content": reply_text})
                except Exception as e:
                    st.error(f"Error querying AI Agent: {e}")
