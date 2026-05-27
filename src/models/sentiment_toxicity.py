import os
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm


class ModelScorer:
    """Scores datasets for sentiment and toxicity using Hugging Face pipelines."""

    def __init__(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        self.device = device or self._determine_device()
        self.dtype = dtype or self._determine_dtype(self.device)
        self._sentiment_pipeline: Optional[Any] = None
        self._toxicity_pipeline: Optional[Any] = None

    def _determine_device(self) -> torch.device:
        """Heuristically determines the best available compute device."""
        if torch.backends.mps.is_available():
            return torch.device('mps')
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    def _determine_dtype(self, device: torch.device) -> torch.dtype:
        """Determines computation precision based on device compatibility."""
        if device.type in ['mps', 'cuda']:
            return torch.float16
        return torch.float32

    def initialize_models(self) -> None:
        """Instantiates HF pipelines for sentiment and toxicity."""
        if self._sentiment_pipeline is not None and self._toxicity_pipeline is not None:
            return

        print(f"Initializing Models on {self.device} with precision {self.dtype}...")
        sentiment_model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self._sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=sentiment_model,
            tokenizer=sentiment_model,
            device=self.device,
            torch_dtype=self.dtype
        )

        toxicity_model = "unitary/toxic-bert"
        self._toxicity_pipeline = pipeline(
            "text-classification",
            model=toxicity_model,
            tokenizer=toxicity_model,
            device=self.device,
            return_all_scores=True,
            torch_dtype=self.dtype
        )

    def _extract_toxic_score(self, pipeline_out: Any) -> float:
        """Parses toxicity pipeline output to extract the specific toxic label score."""
        if isinstance(pipeline_out, list):
            return next(
                (item['score'] for item in pipeline_out if item['label'] == 'toxic'),
                pipeline_out[0]['score'] if pipeline_out else 0.0
            )
        return pipeline_out.get('score', 0.0)

    def score_texts(self, texts: List[str], batch_size: int = 128) -> Tuple[List[str], List[float]]:
        """Scores a list of texts for sentiment labels and toxicity floats."""
        self.initialize_models()
        assert self._sentiment_pipeline is not None
        assert self._toxicity_pipeline is not None

        print(f"Extracting Sentiment on {len(texts)} rows...")
        sentiments = []
        for out in tqdm(
            self._sentiment_pipeline(texts, batch_size=batch_size, truncation=True, max_length=128),
            total=len(texts)
        ):
            sentiments.append(out['label'])

        print("Extracting Toxicity...")
        toxicities = []
        for out in tqdm(
            self._toxicity_pipeline(texts, batch_size=batch_size, truncation=True, max_length=128),
            total=len(texts)
        ):
            toxicities.append(self._extract_toxic_score(out))

        # Enforce safety guardrail overrides
        from models.safety_guardrails import SafetyGuardrail
        guardrail = SafetyGuardrail()
        
        final_sentiments = []
        final_toxicities = []
        for text, sent, tox in zip(texts, sentiments, toxicities):
            mod_sent, mod_tox = guardrail.moderate_score(text, sent, tox)
            final_sentiments.append(mod_sent)
            final_toxicities.append(mod_tox)

        return final_sentiments, final_toxicities

    def score_file(self, input_file: str, output_file: str) -> None:
        """Applies sentiment and toxicity pipeline scoring to a single file."""
        print(f"\nLoading data from {input_file}...")
        try:
            df = pd.read_csv(input_file, low_memory=False)
        except Exception as e:
            print(f"Error reading {input_file}: {e}")
            return

        if 'clean_text' not in df.columns:
            print(f"Skipping {input_file}: No 'clean_text' column found.")
            return

        # Prepare outputs map
        df['clean_text'] = df['clean_text'].fillna("").astype(str).str.strip()
        valid_mask = df['clean_text'] != ""
        valid_texts = df.loc[valid_mask, 'clean_text'].tolist()

        df['sentiment'] = 'neutral'
        df['toxicity'] = 0.0

        if not valid_texts:
            print("No valid non-empty texts to score.")
        else:
            sentiments, toxicities = self.score_texts(valid_texts)
            df.loc[valid_mask, 'sentiment'] = sentiments
            df.loc[valid_mask, 'toxicity'] = toxicities

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        print(f"Saving results to {output_file}...")
        df.to_csv(output_file, index=False)

    def score_directory(self, input_dir: str = "data/processed", output_dir: str = "data/scored") -> None:
        """Scores all CSV files in the input folder that have not been scored yet."""
        if not os.path.exists(input_dir):
            print(f"Error: {input_dir} not found.")
            return

        for root, _, files in os.walk(input_dir):
            for file in files:
                if not file.endswith('.csv'):
                    continue
                input_file = os.path.join(root, file)
                rel_path = os.path.relpath(input_file, input_dir)
                output_file = os.path.join(output_dir, rel_path)

                if os.path.exists(output_file):
                    print(f"Skipping {input_file}, output already exists at {output_file}")
                    continue

                self.score_file(input_file, output_file)


if __name__ == "__main__":
    scorer = ModelScorer()
    scorer.score_directory()
