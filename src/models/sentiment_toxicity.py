import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm
import os

def main():
    print("Loading data...")
    input_path = "data/processed/polarized_sport_final.csv"
    output_path = "data/processed/final_scored_data.csv"
    
    df = pd.read_csv(input_path)
    
    print(f"Dataset size: {len(df)} rows.")
    
    # Ensure no NaN in clean_text
    df['clean_text'] = df['clean_text'].fillna(df['text']).astype(str)
    
    # We will use MPS (Apple Silicon GPU) if available, otherwise CPU
    device = torch.device('mps') if torch.backends.mps.is_available() else (
             torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    device_id = -1 if device.type == 'cpu' else (0 if device.type == 'cuda' else 'mps')
    print(f"Using device: {device}")

    # 1. Sentiment Analysis Pipeline
    print("Initializing Sentiment Analysis model...")
    # cardiffnlp/twitter-roberta-base-sentiment-latest is great for 3-class sentiment
    sentiment_model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_model, device=device)
    
    # 2. Toxicity Detection Pipeline
    print("Initializing Toxicity Detection model...")
    # unitary/toxic-bert outputs toxicity score
    toxicity_model = "unitary/toxic-bert"
    toxicity_pipeline = pipeline("text-classification", model=toxicity_model, tokenizer=toxicity_model, device=device, return_all_scores=True)

    print(f"Running inference on {len(df)} rows. This might take a while...")
    
    # Extract texts as list for pipeline
    texts = df['clean_text'].tolist()
    
    # Process Sentiment
    print("Extracting Sentiment...")
    sentiments = []
    # We use batch size for efficiency
    for out in tqdm(sentiment_pipeline(texts, batch_size=32, truncation=True, max_length=512), total=len(texts)):
        sentiments.append(out['label']) # Typically 'positive', 'neutral', 'negative'
        
    df['sentiment'] = sentiments

    # Process Toxicity
    print("Extracting Toxicity...")
    toxicities = []
    for out in tqdm(toxicity_pipeline(texts, batch_size=32, truncation=True, max_length=512), total=len(texts)):
        # out can be a dict {'label': 'toxic', 'score': ...} or a list of dicts depending on pipeline config
        if isinstance(out, list):
            toxic_score = next((item['score'] for item in out if item['label'] == 'toxic'), out[0]['score'] if out else 0.0)
        else:
            toxic_score = out.get('score', 0.0)
        toxicities.append(toxic_score)
        
    df['toxicity'] = toxicities

    print("Saving results...")
    df.to_csv(output_path, index=False)
    print(f"Scored data saved successfully to {output_path}!")

if __name__ == "__main__":
    main()
