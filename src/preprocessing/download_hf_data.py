import os
from datasets import load_dataset
import pandas as pd

def download_from_hf():
    print("Fetching dataset from Hugging Face...")
    
    # Using a Reddit soccer dataset for more relevant 'Us vs Them' and toxicity analysis
    dataset = load_dataset("singhala/reddit_soccer", split='train') 
    
    # Convert to Pandas DataFrame
    df = pd.DataFrame(dataset)
    
    # We only need the comments, we rename it to 'text' to match the rest of the pipeline
    df = df.rename(columns={'comment': 'text'})
    
    # Drop rows without text
    df = df.dropna(subset=['text'])
    
    # Keep only the 'text' column to simplify, and maybe 'title' as context
    df = df[['text', 'title', 'link_flair']]
    
    # Optional: Take a sample of 30,000 rows to match previous data size
    if len(df) > 2000:
        df = df.sample(n=2000, random_state=42).reset_index(drop=True)
    
    # Save to your project structure
    output_path = "data/raw/hf_sport_data.csv"
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Success! {len(df)} Reddit sport entries saved to {output_path}")

if __name__ == "__main__":
    download_from_hf()