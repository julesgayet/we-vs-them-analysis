import os
from typing import Dict, List, Optional
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from .vectorstore import VectorStoreManager


class AIAssistant:
    """Handles vector search context extraction and LLM interaction for the Chat interface."""

    def __init__(self, vectorstore_manager: Optional[VectorStoreManager] = None) -> None:
        load_dotenv()
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if hf_token:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

        self.vectorstore_manager = vectorstore_manager or VectorStoreManager()
        self._chat_model: Optional[ChatHuggingFace] = None

    def _initialize_llm(self) -> None:
        """Initializes the Hugging Face model pipeline if not already loaded."""
        if self._chat_model is not None:
            return

        llm = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-7B-Instruct",
            task="text-generation",
            max_new_tokens=512,
            temperature=0.3,
            repetition_penalty=1.1
        )
        self._chat_model = ChatHuggingFace(llm=llm)

    def retrieve_context_documents(self, query: str, k: int = 3) -> str:
        """Similarity search in FAISS vector store to get supporting comment examples."""
        try:
            vectorstore = self.vectorstore_manager.load_vector_store()
            if vectorstore is None:
                return "No specific examples found."

            docs = vectorstore.similarity_search(query, k=k)
            if not docs:
                return "No specific examples found."

            return "\n".join([f"- {doc.page_content}" for doc in docs])
        except Exception as e:
            print(f"Error searching vector store: {e}")
            return "No specific examples found."

    def build_stats_context(self, filtered_df: pd.DataFrame) -> str:
        """Formats quantitative metrics from the current filtered dataset for the LLM context."""
        if filtered_df.empty:
            return "No messages currently loaded."

        total_msgs = len(filtered_df)
        pol_pct = filtered_df['is_polarized'].mean() * 100
        avg_tox = filtered_df['toxicity'].mean()

        pol_df = filtered_df[filtered_df['is_polarized'] == True]
        non_pol_df = filtered_df[filtered_df['is_polarized'] == False]
        avg_tox_pol = pol_df['toxicity'].mean() if not pol_df.empty else 0.0
        avg_tox_non_pol = non_pol_df['toxicity'].mean() if not non_pol_df.empty else 0.0

        context = (
            f"GLOBAL STATISTICS CONTEXT:\n"
            f"- Total analyzed messages: {total_msgs}\n"
            f"- Polarization Rate (Us vs Them language): {pol_pct:.1f}%\n"
            f"- Average Toxicity Score: {avg_tox:.3f}\n"
            f"- Average Toxicity for Normal messages: {avg_tox_non_pol:.3f}\n"
            f"- Average Toxicity for Polarized messages: {avg_tox_pol:.3f}\n"
        )
        return context

    def generate_system_prompt(self, stats_context: str, retrieved_docs_text: str) -> str:
        """Assembles the system template using data and retrieved documents."""
        system_prompt = (
            "You are 'Project Shield AI', a professional data analyst assistant.\n"
            "You analyze social media polarization and toxicity. Answer concisely and naturally in a conversational way.\n"
            "Use the following GLOBAL STATISTICS and RETRIEVED EXAMPLES to answer the user accurately.\n"
            "Never make up statistics. If the data provides the answer, use it.\n\n"
            f"{stats_context}\n\n"
            f"RETRIEVED EXAMPLES FROM DATASET:\n"
            f"{retrieved_docs_text}"
        )
        return system_prompt

    def ask(self, prompt: str, history: List[Dict[str, str]], filtered_df: pd.DataFrame) -> str:
        """Sends the question along with chat history and vector search context to the LLM."""
        self._initialize_llm()
        assert self._chat_model is not None

        # Build Context
        stats_context = self.build_stats_context(filtered_df)
        retrieved_docs_text = self.retrieve_context_documents(prompt)
        system_prompt = self.generate_system_prompt(stats_context, retrieved_docs_text)

        # Assemble full message chain
        messages_for_llm = [SystemMessage(content=system_prompt)]

        for msg in history:
            if msg["role"] == "user":
                messages_for_llm.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages_for_llm.append(AIMessage(content=msg["content"]))

        response = self._chat_model.invoke(messages_for_llm)
        return response.content
