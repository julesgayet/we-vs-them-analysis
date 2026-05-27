from collections import Counter
import os
from typing import List, Tuple
import pandas as pd


class EDAAnalyzer:
    """Analyzes processed linguistic data to extract key insights."""

    def __init__(self, data_dir: str = "data/processed") -> None:
        self.data_dir = data_dir

    def load_processed_data(self) -> pd.DataFrame:
        """Walks the directory and merges all processed CSV datasets containing polarized info."""
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
        """Helper to safely read a CSV and append to dfs list if valid."""
        try:
            temp_df = pd.read_csv(filepath, low_memory=False)
            if 'is_polarized' in temp_df.columns and 'clean_text' in temp_df.columns:
                dfs.append(temp_df)
        except Exception as e:
            print(f"Skipping file {filepath} due to error: {e}")

    def get_top_context_words(
        self,
        text_series: pd.Series,
        stop_words: List[str],
        min_length: int = 4,
        top_n: int = 20
    ) -> List[Tuple[str, int]]:
        """Filters short/stop words from text series and counts most common."""
        words = " ".join(text_series.astype(str)).lower().split()
        filtered_words = [
            w for w in words
            if len(w) > min_length and w not in stop_words
        ]
        return Counter(filtered_words).most_common(top_n)

    def run_analysis(self, stop_words: List[str], top_n: int = 20) -> None:
        """Executes the analysis and prints top words."""
        df = self.load_processed_data()
        if df.empty:
            print("Error: No processed data found. Run linguistic_analysis.py first.")
            return

        polarized_df = df[df['is_polarized'] == True]
        if polarized_df.empty:
            print("No polarized comments found to analyze.")
            return

        common_words = self.get_top_context_words(
            polarized_df['clean_text'],
            stop_words=stop_words,
            top_n=top_n
        )
        print("Top words in polarized discussions:", common_words)


if __name__ == "__main__":
    default_stop_words = ["this", "that", "with", "from"]
    analyzer = EDAAnalyzer()
    analyzer.run_analysis(stop_words=default_stop_words)