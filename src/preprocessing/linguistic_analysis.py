import os
import re
from typing import Any, List, Optional
import pandas as pd
import spacy


class TextCleaner:
    """Cleans raw text data for downstream NLP tasks."""

    URL_PATTERN = re.compile(r'http\S+')
    WHITESPACE_PATTERN = re.compile(r'\s+')

    def clean(self, text: Any) -> str:
        """Basic text cleaning: removes URLs and standardizes whitespace."""
        if not isinstance(text, str):
            return ""

        text = self.URL_PATTERN.sub('', text)
        text = self.WHITESPACE_PATTERN.sub(' ', text)
        return text.strip()


class PolarizationAnalyzer:
    """Analyzes text for polarized 'Us vs Them' linguistic structures."""

    def __init__(self, spacy_model_name: str = "en_core_web_sm") -> None:
        # Load spaCy model with only necessary components for speed
        self.nlp = spacy.load(spacy_model_name, disable=["ner", "lemmatizer"])
        self.us_group = {"we", "us", "our", "ours"}
        self.them_group = {"they", "them", "their", "theirs"}

    def analyze_batch(self, texts: List[str], batch_size: int = 200) -> pd.DataFrame:
        """Process a list of texts and return counts and polarization flags."""
        us_counts = []
        them_counts = []

        print(f"🚀 Running NLP pipe on {len(texts)} rows...")
        for doc in self.nlp.pipe(texts, batch_size=batch_size):
            tokens = [t.text.lower() for t in doc]
            us_cnt = sum(1 for t in tokens if t in self.us_group)
            them_cnt = sum(1 for t in tokens if t in self.them_group)

            us_counts.append(us_cnt)
            them_counts.append(them_cnt)

        is_polarized = [
            (u > 0 and t > 0) for u, t in zip(us_counts, them_counts)
        ]

        return pd.DataFrame({
            "us_count": us_counts,
            "them_count": them_counts,
            "is_polarized": is_polarized
        })


class LinguisticPipelineManager:
    """Coordinates the file finding, processing, and saving pipeline."""

    POSSIBLE_COLUMNS = [
        "childCommentText", "text", "Text",
        "video_transcription_text", "caption", "parentText"
    ]

    def __init__(self, cleaner: TextCleaner, analyzer: PolarizationAnalyzer) -> None:
        self.cleaner = cleaner
        self.analyzer = analyzer

    def find_text_column(self, df: pd.DataFrame) -> Optional[str]:
        """Identifies the appropriate column containing text data."""
        for col in self.POSSIBLE_COLUMNS:
            if col in df.columns:
                return col
        return None

    def process_file(self, input_file: str, output_file: str) -> None:
        """Applies cleaning and polarization classification on a single file."""
        print(f"\nProcessing: {input_file}")
        try:
            df = pd.read_csv(input_file, low_memory=False)
        except Exception as e:
            print(f"Error reading {input_file}: {e}")
            return

        text_col = self.find_text_column(df)
        if not text_col:
            print(f"Warning: No valid text column found in {input_file}. Skipping.")
            return

        print(f"Using column '{text_col}' for analysis.")
        df['clean_text'] = df[text_col].apply(self.cleaner.clean)

        analysis_df = self.analyzer.analyze_batch(df['clean_text'].tolist())
        df = pd.concat([df.reset_index(drop=True), analysis_df.reset_index(drop=True)], axis=1)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"✅ Saved to {output_file}")
        print(f"Found {df['is_polarized'].sum()} polarized samples.")

    def run(self, input_dir: str = "data/raw", output_dir: str = "data/processed") -> None:
        """Walks raw directory to process all matching csv files."""
        if not os.path.exists(input_dir):
            print(f"Error: {input_dir} not found.")
            return

        for root, _, files in os.walk(input_dir):
            # Skip sentiment folder as it contains already analyzed data from tutor
            if "sentiment" in root.split(os.sep):
                continue

            for file in files:
                if not file.endswith('.csv'):
                    continue
                input_file = os.path.join(root, file)
                rel_path = os.path.relpath(input_file, input_dir)
                output_file = os.path.join(output_dir, rel_path)
                self.process_file(input_file, output_file)


if __name__ == "__main__":
    cleaner = TextCleaner()
    analyzer = PolarizationAnalyzer()
    manager = LinguisticPipelineManager(cleaner, analyzer)
    manager.run()