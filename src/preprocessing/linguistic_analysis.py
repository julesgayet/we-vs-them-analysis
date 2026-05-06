import pandas as pd
import spacy
import os
import re

# Load spaCy model with only necessary components for speed
nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

def clean_text(text):
    """Basic cleaning before NLP processing."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+', '', text) # Remove URLs
    text = re.sub(r'\s+', ' ', text)    # Remove extra whitespace
    return text.strip()

def run_linguistic_pipeline(input_file="data/raw/hf_sport_data.csv"):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    # Load data
    df = pd.read_csv(input_file)
    df['clean_text'] = df['text'].apply(clean_text)
    
    us_group = {"we", "us", "our", "ours"}
    them_group = {"they", "them", "their", "theirs"}
    
    us_counts = []
    them_counts = []

    print(f"🚀 Processing {len(df)} rows with spaCy nlp.pipe...")
    
    # nlp.pipe is much faster for large datasets
    for doc in nlp.pipe(df['clean_text'], batch_size=100):
        tokens = [t.text.lower() for t in doc]
        us_counts.append(sum(1 for t in tokens if t in us_group))
        them_counts.append(sum(1 for t in tokens if t in them_group))

    # Add results to dataframe
    df['us_count'] = us_counts
    df['them_count'] = them_counts
    # Flag polarized entries (Objective: Detecting "Us vs Them" structures)
    df['is_polarized'] = (df['us_count'] > 0) & (df['them_count'] > 0)

    # Save processed data for Phase 2
    output_path = "data/processed/polarized_sport_final.csv"
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Complete. Dataset saved to {output_path}")
    print(f"Found {df['is_polarized'].sum()} polarized samples.")

if __name__ == "__main__":
    run_linguistic_pipeline()