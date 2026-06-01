import os
from typing import Optional
from datasets import load_dataset
import pandas as pd


class HFDataDownloader:
    """Downloader class for fetching datasets from Hugging Face Hub."""

    def __init__(self, dataset_name: str, split: str = "train") -> None:
        self.dataset_name = dataset_name
        self.split = split

    def download_dataset(self) -> Optional[pd.DataFrame]:
        """Downloads dataset from HF and returns as a DataFrame."""
        print(f"Fetching dataset '{self.dataset_name}' from Hugging Face...")
        try:
            dataset = load_dataset(self.dataset_name, split=self.split)
            return pd.DataFrame(dataset)
        except Exception as e:
            print(f"Error fetching dataset {self.dataset_name}: {e}")
            return None

    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes columns and filters empty rows."""
        if 'comment' in df.columns:
            df = df.rename(columns={'comment': 'text'})
        elif 'commentary' in df.columns:
            df = df.rename(columns={'commentary': 'text'})

        if 'text' not in df.columns:
            print("Warning: Expected 'text', 'comment', or 'commentary' column in dataset. Available columns:", df.columns)
            return df

        # Drop rows without text
        df = df.dropna(subset=['text'])

        # Keep relevant columns
        cols_to_keep = [
            col for col in [
                'text', 'title', 'link_flair', 'event_type', 
                'player_mentioned', 'team', 'league', 'language'
            ] if col in df.columns
        ]
        return df[cols_to_keep]

    def sample_dataset(self, df: pd.DataFrame, sample_size: int = 2000) -> pd.DataFrame:
        """Randomly samples dataset to keep computation lightweight."""
        if len(df) <= sample_size:
            return df
        return df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    def save_dataset(self, df: pd.DataFrame, output_path: str) -> None:
        """Saves the processed DataFrame to CSV."""
        if df.empty:
            print("No data to save.")
            return

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Success! {len(df)} entries saved to {output_path}")

    def run_pipeline(self, output_path: str, sample_size: int = 2000) -> None:
        """Executes full download and preprocessing pipeline."""
        df = self.download_dataset()
        if df is None or df.empty:
            print(f"Pipeline aborted for {self.dataset_name}: dataset could not be retrieved.")
            return

        df = self.preprocess_dataset(df)
        df = self.sample_dataset(df, sample_size)
        self.save_dataset(df, output_path)


if __name__ == "__main__":
    # Download Reddit Soccer Dataset
    reddit_downloader = HFDataDownloader(dataset_name="singhala/reddit_soccer")
    reddit_downloader.run_pipeline(output_path="data/raw/reddit/reddit_soccer.csv", sample_size=2000)

    # Download YallaShoot Football Commentary Dataset
    yallashoot_downloader = HFDataDownloader(dataset_name="yallashoot/football-commentary-dataset")
    yallashoot_downloader.run_pipeline(output_path="data/raw/yallashoot/football_commentary.csv", sample_size=2000)