import pandas as pd
import spacy
import os
import re
from pathlib import Path

# Load spaCy model with only necessary components for speed
nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

def clean_text(text):
    """Basic cleaning before NLP processing."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+', '', text) # Remove URLs
    text = re.sub(r'\s+', ' ', text)    # Remove extra whitespace
    return text.strip()

def find_text_column(df):
    """Dynamically identify the main text column in a dataframe."""
    possible_columns = ["childCommentText", "text", "Text", "video_transcription_text", "caption", "parentText"]
    for col in possible_columns:
        if col in df.columns:
            return col
    return None

def process_file(input_file, output_file):
    print(f"\nProcessing: {input_file}")
    try:
        # Load data
        df = pd.read_csv(input_file, low_memory=False)
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return

    text_col = find_text_column(df)
    if not text_col:
        print(f"Warning: No valid text column found in {input_file}. Skipping.")
        return

    print(f"Using column '{text_col}' for analysis.")
    df['clean_text'] = df[text_col].apply(clean_text)
    
    us_group = {"we", "us", "our", "ours"}
    them_group = {"they", "them", "their", "theirs"}
    
    us_counts = []
    them_counts = []

    print(f"🚀 Running NLP pipe on {len(df)} rows...")
    
    for doc in nlp.pipe(df['clean_text'], batch_size=200):
        tokens = [t.text.lower() for t in doc]
        us_counts.append(sum(1 for t in tokens if t in us_group))
        them_counts.append(sum(1 for t in tokens if t in them_group))

    df['us_count'] = us_counts
    df['them_count'] = them_counts
    df['is_polarized'] = (df['us_count'] > 0) & (df['them_count'] > 0)

    # Save processed data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"✅ Saved to {output_file}")
    print(f"Found {df['is_polarized'].sum()} polarized samples.")

def run_linguistic_pipeline(input_dir="data/raw", output_dir="data/processed"):
    if not os.path.exists(input_dir):
        print(f"Error: {input_dir} not found.")
        return

    for root, dirs, files in os.walk(input_dir):
        # Skip sentiment folder as it contains already analyzed data from tutor
        if "sentiment" in root.split(os.sep):
            continue
            
        for file in files:
            if file.endswith('.csv'):
                input_file = os.path.join(root, file)
                rel_path = os.path.relpath(input_file, input_dir)
                output_file = os.path.join(output_dir, rel_path)
                process_file(input_file, output_file)

if __name__ == "__main__":
    run_linguistic_pipeline()