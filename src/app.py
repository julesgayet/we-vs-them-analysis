import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

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
tab1, tab2, tab3, tab4 = st.tabs(["📊 Global Overview", "🚨 Toxicity Analysis", "🔍 Data Explorer", "🤖 AI Assistant"])

with tab1:
    st.header("Global Data Overview")
    
    # KPIs
    col1, col2, col3 = st.columns(3)
    pol_pct = filtered_df['is_polarized'].mean() * 100
    with col1: st.metric("Analyzed Messages", f"{len(filtered_df):,}")
    with col2: st.metric("Polarization Rate", f"{pol_pct:.1f}%")
    with col3: st.metric("Average Toxicity Score", f"{filtered_df['toxicity'].mean():.3f}")
    
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

    st.markdown("### 🚨 Top 10 Most Toxic Comments")
    top_toxic = filtered_df[filtered_df['is_scored'] == True].sort_values(by='toxicity', ascending=False).head(10)
    
    if top_toxic.empty:
        st.info("No comments have been analyzed yet.")
    else:
        for idx, row in top_toxic.iterrows():
            with st.expander(f"Toxicity: {row['toxicity']:.2f} | {row['platform']} | Sentiment: {row['sentiment']}"):
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
    st.header("🤖 Project Shield AI Assistant")
    st.markdown("Ask the AI agent any question about your data. It analyzes both global statistics and reads specific comments!")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if prompt := st.chat_input("Ask something (e.g. 'What is the mood on Twitter today?')"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Analyzing data and generating response..."):
                try:
                    stats_context = f"""
                    GLOBAL STATISTICS CONTEXT:
                    - Total analyzed messages: {len(filtered_df)}
                    - Polarization Rate (Us vs Them language): {pol_pct:.1f}%
                    - Average Toxicity Score: {filtered_df['toxicity'].mean():.3f}
                    """
                    if not pol_df.empty and not non_pol_df.empty:
                        stats_context += f"- Average Toxicity for Normal messages: {avg_tox_non_pol:.3f}\n"
                        stats_context += f"- Average Toxicity for Polarized messages: {avg_tox_pol:.3f}\n"
                    
                    faiss_path = "data/faiss_index"
                    retrieved_docs_text = "No specific examples found."
                    if os.path.exists(faiss_path):
                        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                        vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
                        docs = vectorstore.similarity_search(prompt, k=3)
                        retrieved_docs_text = "\n".join([f"- {d.page_content}" for d in docs])
                    
                    llm = HuggingFaceEndpoint(
                        repo_id="HuggingFaceH4/zephyr-7b-beta",
                        task="text-generation",
                        max_new_tokens=512,
                        temperature=0.3,
                        repetition_penalty=1.1
                    )
                    
                    chat_model = ChatHuggingFace(llm=llm)
                    
                    system_prompt = f"""You are 'Project Shield AI', a professional data analyst assistant.
You analyze social media polarization and toxicity. Answer concisely and naturally in a conversational way.
Use the following GLOBAL STATISTICS and RETRIEVED EXAMPLES to answer the user accurately.
Never make up statistics. If the data provides the answer, use it.

{stats_context}

RETRIEVED EXAMPLES FROM DATASET:
{retrieved_docs_text}"""
                    
                    # Build conversational history
                    messages_for_llm = [SystemMessage(content=system_prompt)]
                    
                    # Add all previous messages except the current prompt (since it's already in session_state, wait, we appended it above!)
                    # Actually, the current prompt is the LAST item in st.session_state.messages.
                    for msg in st.session_state.messages:
                        if msg["role"] == "user":
                            messages_for_llm.append(HumanMessage(content=msg["content"]))
                        else:
                            messages_for_llm.append(AIMessage(content=msg["content"]))
                    
                    response = chat_model.invoke(messages_for_llm)
                    
                    # The response is an AIMessage object.
                    reply_text = response.content
                    
                    st.markdown(reply_text)
                    st.session_state.messages.append({"role": "assistant", "content": reply_text})
                except Exception as e:
                    st.error(f"Error querying AI Agent: {e}")
