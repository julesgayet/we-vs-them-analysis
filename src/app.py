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
    .stApp { background-color: #f8fafc; color: #0f172a; }
    .main-title {
        font-size: 3.2rem; font-weight: 800;
        background: linear-gradient(90deg, #ff4b4b, #ff8f00);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0px; padding-bottom: 0px;
    }
    .subtitle { color: #64748b; font-size: 1.2rem; margin-bottom: 30px; }
    [data-testid="stMetricValue"] { font-size: 2.5rem !important; font-weight: 800 !important; color: #0f172a !important; }
    [data-testid="stMetricLabel"] { font-size: 1.1rem !important; color: #64748b !important; text-transform: uppercase; font-weight: 600; }
    [data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #e2e8f0; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; font-size: 1.2rem; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { background-color: rgba(255, 75, 75, 0.08) !important; color: #ff4b4b !important; border-bottom: 2px solid #ff4b4b !important;}
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
st.sidebar.title("Controls")

# Scan processed directory dynamically to find all platform names
tutor_platforms = ['Twitter', 'Tiktok', 'Instagram']
hf_platforms = ['Reddit']

all_platforms_in_data = set()
if os.path.exists("data/processed"):
    for entry in os.scandir("data/processed"):
        if entry.is_dir() and entry.name not in ['__pycache__', 'faiss_index']:
            all_platforms_in_data.add(entry.name.capitalize())

dataset_sources = []
if any(p in all_platforms_in_data for p in tutor_platforms):
    dataset_sources.append("Dataset (Twitter, TikTok, Instagram)")
if any(p in all_platforms_in_data for p in hf_platforms):
    dataset_sources.append("Hugging Face Dataset (Reddit)")

# Add custom uploaded platforms dynamically
for plat in sorted(all_platforms_in_data):
    if plat not in tutor_platforms and plat not in hf_platforms:
        dataset_sources.append(f"{plat} Dataset ({plat})")

if not dataset_sources:
    dataset_sources = ["No Datasets Found"]

selected_source = st.sidebar.radio("Select Dataset Source", dataset_sources)
st.session_state.selected_source = selected_source

if selected_source == "Dataset (Twitter, TikTok, Instagram)":
    allowed_platforms = ['Twitter', 'Tiktok', 'Instagram']
elif selected_source == "Hugging Face Dataset (Reddit)":
    allowed_platforms = ['Reddit']
else:
    import re
    match = re.search(r'\((.*?)\)', selected_source)
    allowed_platforms = [match.group(1)] if match else []

# Filter dataset to selected source
df = df[df['platform'].isin(allowed_platforms)]

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

# Delete Dataset Section
st.sidebar.subheader("Delete a Dataset")
delete_target = st.sidebar.selectbox("Select Dataset to Delete", ["Select..."] + sorted(list(all_platforms_in_data)))
if delete_target != "Select...":
    confirm_delete = st.sidebar.checkbox(f"Confirm deletion of {delete_target}")
    if confirm_delete:
        if st.sidebar.button(f"Permanently Delete {delete_target}", type="primary"):
            with st.sidebar.status(f"Deleting {delete_target} dataset...", expanded=True) as status:
                import shutil
                target_lower = delete_target.lower()
                
                processed_path = os.path.join("data/processed", target_lower)
                scored_path = os.path.join("data/scored", target_lower)
                
                deleted_any = False
                if os.path.exists(processed_path):
                    shutil.rmtree(processed_path)
                    deleted_any = True
                if os.path.exists(scored_path):
                    shutil.rmtree(scored_path)
                    deleted_any = True
                
                if deleted_any:
                    st.write("Rebuilding FAISS vector index...")
                    try:
                        from rag.vectorstore import VectorStoreManager
                        v_manager = VectorStoreManager()
                        v_manager.build_vector_store()
                    except Exception as e:
                        st.write(f"Warning: Failed to rebuild FAISS index: {e}")
                    
                    st.write("Updating Causal Analysis...")
                    try:
                        from preprocessing.causal_analysis import CausalAnalyzer
                        c_analyzer = CausalAnalyzer()
                        c_analyzer.run_analysis()
                    except Exception as e:
                        st.write(f"Warning: Failed to run causal analysis: {e}")
                    
                    status.update(label="✅ Success! Dataset Deleted & Re-indexed", state="complete")
                    st.sidebar.success(f"Deleted {delete_target} dataset. Reloading...")
                    st.cache_data.clear()
                    try:
                        st.rerun()
                    except AttributeError:
                        st.experimental_rerun()
                else:
                    status.update(label="❌ Dataset files not found", state="error")

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
            fig_plat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#0f172a")
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
            fig_sent.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#0f172a", showlegend=False)
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
            x=["Normal", "Polarized ('We vs Them')"],
            y=[avg_tox_non_pol, avg_tox_pol],
            color=["Normal", "Polarized"],
            color_discrete_map={"Normal": "#1f77b4", "Polarized": "#ff4b4b"},
            labels={'x': 'Message Type', 'y': 'Average Toxicity Score'},
            title="Direct Comparison: Impact of polarized language on toxicity"
        )
        fig_tox.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#0f172a", showlegend=False)
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
    
    # Calculate dynamic fairness metrics based on the active df (representing selected_source)
    if not df.empty and 'is_polarized' in df.columns and 'toxicity' in df.columns:
        fair_df = df.copy()
        fair_df['y_true'] = fair_df['is_polarized'].astype(int)
        fair_df['y_pred'] = (fair_df['toxicity'] > 0.5).astype(int)
        
        platforms_list = sorted(list(fair_df['platform'].unique()))
        platform_metrics = {}
        
        for plat in platforms_list:
            plat_df = fair_df[fair_df['platform'] == plat]
            tp = ((plat_df['y_true'] == 1) & (plat_df['y_pred'] == 1)).sum()
            fp = ((plat_df['y_true'] == 0) & (plat_df['y_pred'] == 1)).sum()
            tn = ((plat_df['y_true'] == 0) & (plat_df['y_pred'] == 0)).sum()
            fn = ((plat_df['y_true'] == 1) & (plat_df['y_pred'] == 0)).sum()
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            
            platform_metrics[plat] = {
                "tpr": tpr,
                "fpr": fpr,
                "total_samples": len(plat_df)
            }
        
        tprs = [m['tpr'] for m in platform_metrics.values()]
        fprs = [m['fpr'] for m in platform_metrics.values()]
        
        fpr_gap = max(fprs) - min(fprs) if fprs else 0.0
        tpr_gap = max(tprs) - min(tprs) if tprs else 0.0
        eo_disparity = fpr_gap + tpr_gap
        
        # Format report string dynamically
        report_lines = []
        report_lines.append("==================================================")
        report_lines.append("       DYNAMIC MODEL BIAS & FAIRNESS REPORT       ")
        report_lines.append("==================================================")
        report_lines.append(f"Selected Dataset Source: {selected_source}")
        report_lines.append(f"Total scored comments analyzed: {len(fair_df):,}\n")
        
        report_lines.append("--- Platform Fairness Metrics (Proxy: Polarization as Target) ---")
        for plat, metrics in platform_metrics.items():
            report_lines.append(f"Platform: {plat}")
            report_lines.append(f"  - Total Samples: {metrics['total_samples']:,}")
            report_lines.append(f"  - True Positive Rate (TPR): {metrics['tpr']:.4f}")
            report_lines.append(f"  - False Positive Rate (FPR): {metrics['fpr']:.4f}\n")
        
        report_lines.append("--- Disparities ---")
        report_lines.append(f"  - False Positive Rate (FPR) Gap: {fpr_gap:.4f}")
        report_lines.append(f"  - True Positive Rate (TPR) Gap: {tpr_gap:.4f}")
        report_lines.append(f"  - Equalized Odds (EO) Disparity: {eo_disparity:.4f}\n")
        
        report_lines.append("--- Bias Mitigation Guidelines ---")
        report_lines.append("1. Platform-Specific Classification Thresholds:")
        report_lines.append("   Adjust toxicity classification thresholds per platform to calibrate and equalize FPRs.")
        report_lines.append("2. Targeted Data Augmentation:")
        report_lines.append("   Collect more training examples from under-represented platforms (e.g. TikTok) to align model representations.")
        report_lines.append("3. Dialect/Slang Alignment:")
        report_lines.append("   Fine-tune toxicity models on domain-specific social media text to decrease false positives caused by benign in-group slang.")
        report_lines.append("4. Regular Audits:")
        report_lines.append("   Run continuous fairness pipelines on newly collected samples to monitor drift in EO disparity.")
        
        dynamic_report = "\n".join(report_lines)
        st.text_area("Fairness Report Details (Dynamic)", dynamic_report, height=450)
    else:
        st.warning("No data available to calculate fairness metrics.")
            
    st.markdown("---")
    st.subheader("Detected Temporal Activity Spikes")
    
    # Dynamically detect spikes on the active filtered_df
    def detect_dynamic_spikes(f_df: pd.DataFrame, threshold_std: float = 1.5) -> pd.DataFrame:
        if f_df.empty:
            return pd.DataFrame()
        col = 'timestamp' if 'timestamp' in f_df.columns else ('Timestamp' if 'Timestamp' in f_df.columns else 'createTimeISO')
        if col not in f_df.columns:
            return pd.DataFrame()
        df_dates = f_df.copy()
        df_dates['parsed_date'] = pd.to_datetime(df_dates[col], errors='coerce').dt.date
        df_dates = df_dates.dropna(subset=['parsed_date'])
        if df_dates.empty:
            return pd.DataFrame()
        daily_counts = df_dates.groupby('parsed_date').size().rename('comment_count').reset_index()
        daily_counts = daily_counts.sort_values(by='parsed_date').reset_index(drop=True)
        if len(daily_counts) < 7:
            mean_val = daily_counts['comment_count'].mean()
            std_val = daily_counts['comment_count'].std() if daily_counts['comment_count'].std() > 0 else 1.0
        else:
            rolling = daily_counts['comment_count'].rolling(window=7, min_periods=1)
            mean_val = rolling.mean()
            std_val = rolling.std().fillna(1.0)
        daily_counts['is_spike'] = daily_counts['comment_count'] > (mean_val + threshold_std * std_val)
        return daily_counts[daily_counts['is_spike'] == True]

    spikes_df = detect_dynamic_spikes(filtered_df)
    
    if not spikes_df.empty:
        # Load pre-calculated spikes for event mapping
        spikes_path = "data/processed/detected_spikes.csv"
        event_map = {}
        if os.path.exists(spikes_path):
            try:
                static_spikes = pd.read_csv(spikes_path)
                event_map = dict(zip(static_spikes['parsed_date'].astype(str), static_spikes['event']))
            except Exception:
                pass
                
        event_calendar = {
            "2025-04-15": "Champions League Quarter-Final (PSG vs Barcelona / Dortmund vs Atletico)",
            "2025-04-16": "Champions League Quarter-Final (Man City vs Real Madrid / Bayern vs Arsenal)",
            "2025-04-14": "Premier League Matchday / UCL Match Eve Anticipation",
            "2023-01-16": "Supercopa de España Final (Real Madrid vs Barcelona)",
            "2023-01-19": "Riyadh Season Cup (PSG vs Riyadh XI - Messi vs Ronaldo)",
            "2025-02-25": "Champions League Round of 16 Matches",
            "2025-02-14": "Valentine's Day / European League Fixtures",
            "2025-02-10": "Premier League Monday Night Football",
            "2022-04-28": "Europa League Semi-Finals First Leg",
            "2025-03-04": "Champions League Round of 16 Second Leg"
        }
        
        def get_event(date_val):
            d_str = str(date_val)
            if d_str in event_map:
                return event_map[d_str]
            return event_calendar.get(d_str, "Unidentified Event")
            
        spikes_df['event'] = spikes_df['parsed_date'].apply(get_event)
        
        # Filter to keep only football-related events (since the project is football-focused)
        football_keywords = [
            "champions league", "europa league", "premier league", "laliga", "serie a", "ligue 1",
            "ucl", "uel", "psg", "barcelona", "barca", "real madrid", "madrid", "bayern", 
            "dortmund", "atletico", "man city", "chelsea", "liverpool", "arsenal", "juventus", 
            "milan", "inter", "roma", "tottenham", "spurs", "manchester", "united",
            "messi", "ronaldo", "mbappe", "neymar", "haaland", "lamine", "yamal",
            "supercopa", "copa", "football", "soccer", "foot", "matchday", "el clasico", "clasico",
            "riyadh season cup", "leicester", "leeds", "everton", "newcastle",
            "fixture", "fixtures", "derby", "cup", "tournament"
        ]
        
        filtered_spikes = spikes_df[spikes_df['event'].str.lower().apply(lambda x: any(kw in str(x) for kw in football_keywords))]
        
        if not filtered_spikes.empty:
            st.dataframe(filtered_spikes[['parsed_date', 'comment_count', 'event']], use_container_width=True, height=250)
        else:
            st.info("No football-related spikes detected for the selected platform/dataset source.")
    else:
        st.info("No activity spikes detected for the selected platform/dataset source.")

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
                    
                    # Calculate overall toxicity score
                    probs = st.session_state.model_explainer.predict_probabilities([input_text])
                    toxicity_score = float(probs[0][1])
                    
                    st.markdown("### Model Predictions")
                    col_score, col_status = st.columns(2)
                    with col_score:
                        st.metric("Overall Toxicity Score", f"{toxicity_score:.4f}")
                    with col_status:
                        if toxicity_score >= 0.5:
                            st.error("Model Classification: TOXIC 🚨")
                        else:
                            st.success("Model Classification: SAFE ✅")
                    
                    st.markdown("### Feature Importance Heatmap")
                    st.components.v1.html(heatmap_html, height=185, scrolling=False)
                    
                    st.markdown("### Semantically Similar Comments in FAISS Vector Store")
                    allowed_plat_list = list(filtered_df['platform'].unique()) if not filtered_df.empty else None
                    similar_docs = st.session_state.ai_assistant.retrieve_context_documents(input_text, k=4, allowed_platforms=allowed_plat_list)
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
                    # Build live dataset context for the agent
                    platform_breakdown = df['platform'].value_counts().to_dict()
                    breakdown_str = ", ".join([f"{k}: {v} comments" for k, v in platform_breakdown.items()])
                    
                    dataset_context = (
                        f"- Selected Dataset Source Name: {selected_source}\n"
                        f"- Platforms in this source: {', '.join(allowed_platforms)}\n"
                        f"- Active platform filter in dashboard: {selected_platform}\n"
                        f"- Total messages/comments analyzed: {len(df):,}\n"
                        f"- Comments breakdown per platform: {breakdown_str}\n"
                        f"- Overall polarization rate: {df['is_polarized'].mean() * 100:.2f}%\n"
                        f"- Average toxicity score: {df['toxicity'].mean():.4f}\n"
                    )

                    reply_text = st.session_state.react_agent.run(
                        prompt,
                        chat_history=st.session_state.messages,
                        dataset_context=dataset_context
                    )
                    st.markdown(reply_text)
                    st.session_state.messages.append({"role": "assistant", "content": reply_text})
                except Exception as e:
                    st.error(f"Error querying AI Agent: {e}")

