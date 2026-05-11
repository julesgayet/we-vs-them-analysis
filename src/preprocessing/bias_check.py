import pandas as pd
import matplotlib.pyplot as plt
import os

# Load all processed data
dfs = []
for root, _, files in os.walk("data/processed"):
    for file in files:
        if file.endswith(".csv"):
            try:
                temp_df = pd.read_csv(os.path.join(root, file), low_memory=False)
                if 'clean_text' in temp_df.columns and 'is_polarized' in temp_df.columns:
                    # Determine platform from the folder name
                    platform = os.path.basename(root).capitalize()
                    temp_df['platform'] = platform
                    dfs.append(temp_df)
            except Exception as e:
                pass

if not dfs:
    print("Error: No processed data found. Run linguistic_analysis.py first.")
    exit()

df = pd.concat(dfs, ignore_index=True)

# Calculate distribution by platform
platform_stats = df['platform'].value_counts(normalize=True) * 100
print("--- Dataset Distribution by Platform ---")
print(platform_stats)
print("\n")

# Calculate Polarization by Platform
polarized_stats = df.groupby('platform')['is_polarized'].mean() * 100
print("--- Percentage of Polarized Content by Platform ---")
print(polarized_stats)

# Save visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Distribution of Platform
platform_stats.plot(kind='bar', color='skyblue', ax=ax1)
ax1.set_title("Distribution of Data by Platform")
ax1.set_ylabel("Percentage of Dataset (%)")
ax1.tick_params(axis='x', rotation=45)

# Plot 2: Polarization by Platform
polarized_stats.plot(kind='bar', color='salmon', ax=ax2)
ax2.set_title("Percentage of 'Us vs Them' Content")
ax2.set_ylabel("% Polarized")
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
output_path = "data/processed/platform_distribution.png"
plt.savefig(output_path)
print(f"\n✅ Saved new visualization to {output_path}")