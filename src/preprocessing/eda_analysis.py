import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import os

# Load all processed data
dfs = []
for root, _, files in os.walk("data/processed"):
    for file in files:
        if file.endswith(".csv"):
            try:
                temp_df = pd.read_csv(os.path.join(root, file), low_memory=False)
                if 'is_polarized' in temp_df.columns and 'clean_text' in temp_df.columns:
                    dfs.append(temp_df)
            except Exception as e:
                pass

if not dfs:
    print("Error: No processed data found. Run linguistic_analysis.py first.")
    exit()

df = pd.concat(dfs, ignore_index=True)
polarized_df = df[df['is_polarized'] == True]

def get_top_context_words(text_series, stop_words):
    words = " ".join(text_series.astype(str)).lower().split()
    # Filter out very short words and basic stop words
    filtered_words = [w for w in words if len(w) > 3 and w not in stop_words]
    return Counter(filtered_words).most_common(20)

# Quick visualization of the most common words in polarized comments
common_words = get_top_context_words(polarized_df['clean_text'], ["this", "that", "with", "from"])
print("Top words in polarized discussions:", common_words)