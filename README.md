# We vs Them Analysis Dashboard

This project is a modular data science and Natural Language Processing (NLP) platform designed to monitor, analyze, and quantify **polarization** ("We vs Them" discourse) and **toxicity** across multiple social media networks (Twitter, TikTok, Instagram, and Reddit).

It features an interactive **Streamlit** dashboard integrated with an AI agent powered by **RAG (Retrieval-Augmented Generation)** to query live statistics and retrieve concrete comment examples.

---

## 🛠️ Project Architecture

The codebase is highly modularized, adhering strictly to SOLID principles:

- **`src/preprocessing/`**: Collection and initial linguistic pipelines.
  - `linguistic_analysis.py`: Analyzes grammar structures with **spaCy** to tag polarization.
  - `eda_analysis.py`: Extracts top key phrases in polarized discussions.
  - `bias_check.py`: Calculates platform distributions and exports bias charts.
- **`src/models/`**: Classification and scoring services.
  - `sentiment_toxicity.py`: Scores toxicity (**Toxic-BERT**) and sentiments (**Twitter-RoBERTa**) in batches.
- **`src/rag/`**: Vector database and AI agent logic.
  - `vectorstore.py`: Indexes the combined dataset into a local **FAISS** vector store using SentenceTransformers.
  - `assistant.py`: AI chat agent powered by **Qwen2.5-7B-Instruct** using LangChain.
- **`src/app.py`**: Presentation layer containing the Streamlit web layout.

---

## 🔑 Environment Variables Setup (`.env`)

To run the AI agent and query the models, you must create a file named `.env` at the root of the project.

Copy and fill out the following template:

```env
# Hugging Face Access Token - Required for scoring models and AI Chat
# Generate a token (scope: read or inference) at https://huggingface.co/settings/tokens
HUGGINGFACE_TOKEN=hf_your_huggingface_access_token
```

---

## 🚀 Getting Started

### 1. Clone and Install Dependencies

```bash
# Clone the repository
git clone https://github.com/julesgayet/we-vs-them-analysis
cd we-vs-them-analysis

# Create a Python virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows

# Install required packages
pip install -r requirements.txt

# Download required linguistic model for spaCy NLP tagging
python -m spacy download en_core_web_sm
```

### 2. Run the Data Pipeline

To build or refresh the data displayed in the dashboard, run the scripts in the following order:

```bash
# Ensure virtual environment is active
source venv/bin/activate

# 1. Run linguistic analysis to identify polarized comments
python src/preprocessing/linguistic_analysis.py

# 2. Score toxicity and sentiments
python src/models/sentiment_toxicity.py

# 3. Generate the FAISS vector database for the AI assistant
python src/rag/vectorstore.py
```

### 3. Launch the Streamlit Dashboard

Once the data pipeline runs successfully, start the dashboard:

```bash
streamlit run src/app.py
```

The web application will open automatically in your browser at `http://localhost:8501`.

---

## 📊 Dashboard Modules

- **Global Overview**: Key metrics (polarization rates, message count, average toxicity) and interactive visual breakdowns (Plotly).
- **Toxicity Analysis**: Side-by-side comparison of normal vs. polarized comments, and top 10 most toxic comments list.
- **Data Explorer**: High-performance tabular search with filters (toxicity threshold, platform, sentiment, and polarization).
- **Fairness & Causal**: Platform bias reports (True/False Positive Rate gaps, Equalized Odds) and timeline spike mapping to high-profile football matchdays.
- **XAI Explainer**: Token-level feature importance heatmaps powered by LIME/SHAP to explain deep transformer classification decisions.
- **AI Assistant**: Conversational agent answering complex analytical questions using dynamic stats and retrieved context examples.
