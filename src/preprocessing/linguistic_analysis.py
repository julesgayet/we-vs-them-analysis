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
    """Analyzes text for polarized 'Us vs Them' structures and grammatical features."""

    def __init__(self, spacy_model_name: str = "en_core_web_sm") -> None:
        self.nlp = spacy.load(spacy_model_name, disable=["ner"])
        self.us_group = {"we", "us", "our", "ours"}
        self.them_group = {"they", "them", "their", "theirs"}
        self.intensifiers = {
            "very", "extremely", "totally", "completely", "absolutely", "really", "so", 
            "quite", "hugely", "incredibly", "highly", "fully", "utterly", "tremendously", 
            "remarkably", "substantially", "particularly", "deeply", "strongly", "severely",
            "especially", "wonderfully", "dreadfully", "exceptionally", "terribly", "unusually"
        }
        self.hedges = {
            "maybe", "seem", "seemed", "seems", "believe", "believes", "believed",
            "probably", "possibly", "perhaps", "might", "could", "would", "somewhat", 
            "tend", "tends", "tended", "guess", "guesses", "guessed", "suppose", 
            "supposed", "supposes", "think", "thinks", "thought", "appear", "appears", 
            "appeared", "likely", "unlikely", "suggest", "suggests", "suggested", 
            "doubt", "doubts", "doubted", "may", "almost", "mostly", "apparently", 
            "arguably", "mainly", "generally"
        }

    def count_us_group(self, doc: Any) -> int:
        """Counts 'Us' pronouns in the document."""
        return sum(1 for t in doc if t.text.lower() in self.us_group)

    def count_them_group(self, doc: Any) -> int:
        """Counts 'Them' pronouns in the document."""
        return sum(1 for t in doc if t.text.lower() in self.them_group)

    def count_negations(self, doc: Any) -> int:
        """Counts grammatical negation dependency labels."""
        return sum(1 for t in doc if t.dep_ == "neg")

    def count_intensifiers(self, doc: Any) -> int:
        """Counts modifying adverbs from the intensifier dictionary."""
        return sum(1 for t in doc if t.pos_ == "ADV" and t.text.lower() in self.intensifiers)

    def count_hedges(self, doc: Any) -> int:
        """Counts hedge markers from the hedge dictionary."""
        return sum(1 for t in doc if t.text.lower() in self.hedges)

    def analyze_batch(self, texts: List[str], batch_size: int = 200) -> pd.DataFrame:
        """Process a list of texts and return counts and polarization flags."""
        us_counts = []
        them_counts = []
        neg_counts = []
        int_counts = []
        hedge_counts = []
        is_polarized = []

        print(f"🚀 Running NLP pipe on {len(texts)} rows...")
        for doc in self.nlp.pipe(texts, batch_size=batch_size):
            us_cnt = self.count_us_group(doc)
            them_cnt = self.count_them_group(doc)
            neg_cnt = self.count_negations(doc)
            int_cnt = self.count_intensifiers(doc)
            hedge_cnt = self.count_hedges(doc)

            us_counts.append(us_cnt)
            them_counts.append(them_cnt)
            neg_counts.append(neg_cnt)
            int_counts.append(int_cnt)
            hedge_counts.append(hedge_cnt)
            is_polarized.append((us_cnt > 0 and them_cnt > 0) or ((us_cnt > 0 or them_cnt > 0) and (int_cnt > 0 or neg_cnt > 0)))

        return pd.DataFrame({
            "us_count": us_counts,
            "them_count": them_counts,
            "negation_count": neg_counts,
            "intensifier_count": int_counts,
            "hedge_count": hedge_counts,
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
                
                if os.path.exists(output_file):
                    print(f"Skipping {input_file}, output already exists at {output_file}")
                    continue
                    
                self.process_file(input_file, output_file)



if __name__ == "__main__":
    cleaner = TextCleaner()
    analyzer = PolarizationAnalyzer()
    manager = LinguisticPipelineManager(cleaner, analyzer)
    manager.run()