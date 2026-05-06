import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Load your processed data
df = pd.read_csv("data/processed/polarized_sport_final.csv")
polarized_df = df[df['is_polarized'] == True]

def get_top_context_words(text_series, stop_words):
    words = " ".join(text_series).lower().split()
    # Filter out very short words and basic stop words
    filtered_words = [w for w in words if len(w) > 3 and w not in stop_words]
    return Counter(filtered_words).most_common(20)

# Quick visualization of the most common words in polarized comments
common_words = get_top_context_words(polarized_df['text'], ["this", "that", "with", "from"])
print("Top words in polarized discussions:", common_words)