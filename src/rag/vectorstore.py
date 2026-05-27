import os
import time
from typing import List, Optional
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


class VectorStoreManager:
    """Manages the creation, persistence, and loading of the FAISS vector index."""

    def __init__(
        self,
        input_dir: str = "data/scored",
        vectorstore_path: str = "data/faiss_index",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> None:
        self.input_dir = input_dir
        self.vectorstore_path = vectorstore_path
        self.embedding_model_name = embedding_model_name

    def load_scored_datasets(self) -> pd.DataFrame:
        """Loads and combines all scored CSV datasets from the input directory."""
        if not os.path.exists(self.input_dir):
            print(f"Scored input directory not found: {self.input_dir}")
            return pd.DataFrame()

        dfs = []
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if not file.endswith('.csv'):
                    continue
                self._load_single_csv(os.path.join(root, file), dfs)

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)

    def _load_single_csv(self, filepath: str, dfs: List[pd.DataFrame]) -> None:
        """Safely loads a single CSV file, cleaning up empty text rows."""
        try:
            df = pd.read_csv(filepath, low_memory=False)
            df = df[df['clean_text'].notna() & (df['clean_text'].str.strip() != "")]
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    def filter_and_sample(
        self,
        df: pd.DataFrame,
        toxicity_threshold: float = 0.4,
        normal_sample_size: int = 5000,
        embed_all: bool = True
    ) -> pd.DataFrame:
        """Filters highly polarized/toxic comments and samples normal comments, or returns all."""
        if df.empty:
            return df

        full_df = df.drop_duplicates(subset=['clean_text'])
        if embed_all:
            return full_df

        # Polarized or highly toxic comments
        polarized_or_toxic = full_df[
            (full_df['is_polarized'] == True) | (full_df['toxicity'] > toxicity_threshold)
        ]

        # Normal comments
        normal = full_df[
            (full_df['is_polarized'] == False) & (full_df['toxicity'] <= toxicity_threshold)
        ]

        sample_size = min(normal_sample_size, len(normal))
        sampled_normal = normal.sample(n=sample_size, random_state=42) if sample_size > 0 else normal

        target_df = pd.concat([polarized_or_toxic, sampled_normal]).drop_duplicates(subset=['clean_text'])
        return target_df

    def build_vector_store(
        self,
        toxicity_threshold: float = 0.4,
        normal_sample_size: int = 5000,
        embed_all: bool = True
    ) -> None:
        """Extracts text, loads embeddings, creates FAISS vector index, and saves it."""
        df = self.load_scored_datasets()
        if df.empty:
            print("No scored data found. Vector store generation aborted.")
            return

        target_df = self.filter_and_sample(
            df, 
            toxicity_threshold=toxicity_threshold, 
            normal_sample_size=normal_sample_size, 
            embed_all=embed_all
        )
        print(f"Embedding {len(target_df)} documents for RAG...")

        loader = DataFrameLoader(target_df, page_content_column="clean_text")
        docs = loader.load()

        print(f"Loading HuggingFaceEmbeddings ({self.embedding_model_name})...")
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

        start_time = time.time()
        print("Building FAISS index (this may take a few minutes)...")
        vectorstore = FAISS.from_documents(docs, embeddings)

        os.makedirs(os.path.dirname(self.vectorstore_path), exist_ok=True)
        vectorstore.save_local(self.vectorstore_path)
        print(f"FAISS index saved successfully to {self.vectorstore_path} in {time.time() - start_time:.1f} seconds!")

    def load_vector_store(self, allow_dangerous_deserialization: bool = True) -> Optional[FAISS]:
        """Loads and returns the locally saved FAISS index."""
        if not os.path.exists(self.vectorstore_path):
            print(f"Vector store path does not exist: {self.vectorstore_path}")
            return None

        print(f"Loading FAISS index from {self.vectorstore_path}...")
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        return FAISS.load_local(
            self.vectorstore_path,
            embeddings,
            allow_dangerous_deserialization=allow_dangerous_deserialization
        )


if __name__ == "__main__":
    manager = VectorStoreManager()
    manager.build_vector_store()
