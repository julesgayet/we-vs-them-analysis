import pandas as pd
import matplotlib.pyplot as plt

# Load your processed data
df = pd.read_csv("data/processed/polarized_sport_final.csv")

# Define categories for bias check
categories = {
    'Football/Soccer': ['football', 'soccer', 'fifa', 'goal', 'pitch'],
    'Basketball': ['nba', 'basketball', 'hoop', 'lakers'],
    'Olympics': ['olympic', 'gold medal', 'athens', 'games'],
    'Baseball': ['baseball', 'mlb', 'sox', 'yankees']
}

def detect_sport(text):
    text = str(text).lower()
    for sport, keywords in categories.items():
        if any(kw in text for kw in keywords):
            return sport
    return 'Other/General'

# Apply categorization
df['sport_category'] = df['text'].apply(detect_sport)

# Calculate distribution
stats = df['sport_category'].value_counts(normalize=True) * 100
print("--- Dataset Distribution (Bias Check) ---")
print(stats)

# Save visualization for your internship report
plt.figure(figsize=(8, 8)) # size of the graph
stats.plot(kind='bar', color='skyblue')
plt.title("Distribution of Sports in Dataset")
plt.ylabel("Percentage (%)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("data/processed/bias_distribution.png")