import os
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import time

def build_index():
    input_dir = "data/scored"
    vectorstore_path = "data/faiss_index"
    
    print("Gathering scored datasets...")
    dfs = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(root, file), low_memory=False)
                # Keep only valid text
                df = df[df['clean_text'].notna() & (df['clean_text'].str.strip() != "")]
                dfs.append(df)
                
    if not dfs:
        print("No scored data found in data/scored/")
        return
        
    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.drop_duplicates(subset=['clean_text'])
    
    # To keep inference very fast for the vector DB generation, 
    # we'll embed polarized comments and highly toxic comments, and a random sample of normal ones
    # (Since the goal is to investigate 'Us vs Them' and toxicity)
    
    pol_or_tox = full_df[(full_df['is_polarized'] == True) | (full_df['toxicity'] > 0.4)]
    normal = full_df[(full_df['is_polarized'] == False) & (full_df['toxicity'] <= 0.4)].sample(n=min(5000, len(full_df)), random_state=42)
    
    target_df = pd.concat([pol_or_tox, normal]).drop_duplicates(subset=['clean_text'])
    print(f"Embedding {len(target_df)} highly relevant documents for RAG (Polarized + Toxic + Sample of Normal)...")
    
    loader = DataFrameLoader(target_df, page_content_column="clean_text")
    docs = loader.load()
    
    print("Loading HuggingFaceEmbeddings (all-MiniLM-L6-v2)...")
    # This runs locally and is free
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    start_time = time.time()
    print("Building FAISS index (this may take a few minutes)...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    os.makedirs(os.path.dirname(vectorstore_path), exist_ok=True)
    vectorstore.save_local(vectorstore_path)
    
    print(f"FAISS index saved successfully to {vectorstore_path} in {time.time() - start_time:.1f} seconds!")

if __name__ == "__main__":
    build_index()
