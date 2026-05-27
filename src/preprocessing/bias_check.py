import os
from typing import List, Tuple
import matplotlib.pyplot as plt
import pandas as pd


class BiasAnalyzer:
    """Analyzes datasets for platform distribution and polarization bias."""

    def __init__(self, data_dir: str = "data/processed") -> None:
        self.data_dir = data_dir

    def load_processed_data(self) -> pd.DataFrame:
        """Loads and combines all processed datasets, appending platform column."""
        if not os.path.exists(self.data_dir):
            print(f"Directory not found: {self.data_dir}")
            return pd.DataFrame()

        dfs = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if not file.endswith(".csv"):
                    continue
                self._load_single_file(os.path.join(root, file), dfs)

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)

    def _load_single_file(self, filepath: str, dfs: List[pd.DataFrame]) -> None:
        """Helper to load csv and assign platform based on folder structure."""
        try:
            temp_df = pd.read_csv(filepath, low_memory=False)
            if 'clean_text' in temp_df.columns and 'is_polarized' in temp_df.columns:
                rel_path = os.path.relpath(filepath, self.data_dir)
                path_parts = os.path.normpath(rel_path).split(os.sep)
                platform = path_parts[0].capitalize() if len(path_parts) > 1 else "General"
                temp_df['platform'] = platform
                dfs.append(temp_df)
        except Exception as e:
            print(f"Skipping file {filepath} due to error: {e}")

    def compute_stats(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculates distribution statistics and polarization percentages."""
        platform_stats = df['platform'].value_counts(normalize=True) * 100
        polarized_stats = df.groupby('platform')['is_polarized'].mean() * 100
        return platform_stats, polarized_stats

    def save_visualization(
        self,
        platform_stats: pd.Series,
        polarized_stats: pd.Series,
        output_path: str
    ) -> None:
        """Generates and saves matplotlib plots for statistics."""
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
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"\n✅ Saved new visualization to {output_path}")
        plt.close()

    def run_bias_check(self, output_path: str = "data/processed/platform_distribution.png") -> None:
        """Executes the complete bias analysis pipeline."""
        df = self.load_processed_data()
        if df.empty:
            print("Error: No processed data found. Run linguistic_analysis.py first.")
            return

        platform_stats, polarized_stats = self.compute_stats(df)

        print("--- Dataset Distribution by Platform ---")
        print(platform_stats)
        print("\n--- Percentage of Polarized Content by Platform ---")
        print(polarized_stats)

        self.save_visualization(platform_stats, polarized_stats, output_path)


if __name__ == "__main__":
    analyzer = BiasAnalyzer()
    analyzer.run_bias_check()