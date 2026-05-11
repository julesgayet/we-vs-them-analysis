import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm
import os
from pathlib import Path

def process_file(input_file, output_file, sentiment_pipeline, toxicity_pipeline):
    print(f"\nLoading data from {input_file}...")
    try:
        df = pd.read_csv(input_file, low_memory=False)
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return
    
    if 'clean_text' not in df.columns:
        print(f"Skipping {input_file}: No 'clean_text' column found.")
        return

    print(f"Dataset size: {len(df)} rows.")
    
    # Ensure all elements are strictly valid strings
    df['clean_text'] = df['clean_text'].fillna("").astype(str)
    # Force str conversion and drop empty strings or purely whitespace
    df['clean_text'] = df['clean_text'].apply(lambda x: str(x).strip())
    
    # We can only score non-empty texts to prevent pipeline errors
    # We will score all, but map empty texts to neutral/0.0
    valid_mask = df['clean_text'] != ""
    valid_texts = df.loc[valid_mask, 'clean_text'].tolist()
    
    # Prepare output arrays
    sentiments = ['neutral'] * len(df)
    toxicities = [0.0] * len(df)

    if len(valid_texts) > 0:
        print(f"Extracting Sentiment on {len(valid_texts)} valid rows...")
        valid_sentiments = []
        for out in tqdm(sentiment_pipeline(valid_texts, batch_size=128, truncation=True, max_length=128), total=len(valid_texts)):
            valid_sentiments.append(out['label'])
            
        print("Extracting Toxicity...")
        valid_toxicities = []
        for out in tqdm(toxicity_pipeline(valid_texts, batch_size=128, truncation=True, max_length=128), total=len(valid_texts)):
            if isinstance(out, list):
                toxic_score = next((item['score'] for item in out if item['label'] == 'toxic'), out[0]['score'] if out else 0.0)
            else:
                toxic_score = out.get('score', 0.0)
            valid_toxicities.append(toxic_score)
            
        # Re-assign back to full dataframe
        df.loc[valid_mask, 'sentiment'] = valid_sentiments
        df.loc[valid_mask, 'toxicity'] = valid_toxicities
    else:
        df['sentiment'] = sentiments
        df['toxicity'] = toxicities

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"Saving results to {output_file}...")
    df.to_csv(output_file, index=False)

def main():
    input_dir = "data/processed"
    output_dir = "data/scored"
    
    if not os.path.exists(input_dir):
        print(f"Error: {input_dir} not found.")
        return

    device = torch.device('mps') if torch.backends.mps.is_available() else (
             torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"Using device: {device}")

    # Use float16 for massive speedup on modern GPUs/Apple Silicon
    dtype = torch.float16 if device.type in ['mps', 'cuda'] else torch.float32

    print(f"Initializing Models with precision {dtype}...")
    sentiment_model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_model, device=device, torch_dtype=dtype)
    
    toxicity_model = "unitary/toxic-bert"
    toxicity_pipeline = pipeline("text-classification", model=toxicity_model, tokenizer=toxicity_model, device=device, return_all_scores=True, torch_dtype=dtype)

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv'):
                input_file = os.path.join(root, file)
                rel_path = os.path.relpath(input_file, input_dir)
                output_file = os.path.join(output_dir, rel_path)
                
                if os.path.exists(output_file):
                    print(f"Skipping {input_file}, output already exists at {output_file}")
                    continue
                    
                process_file(input_file, output_file, sentiment_pipeline, toxicity_pipeline)

if __name__ == "__main__":
    main()
