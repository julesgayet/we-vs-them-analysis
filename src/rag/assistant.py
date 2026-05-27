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

    def retrieve_context_documents(self, query: str, k: int = 6) -> str:
        """Similarity search in FAISS vector store to get supporting comment examples with metadata."""
        try:
            vectorstore = self.vectorstore_manager.load_vector_store()
            if vectorstore is None:
                return "No specific examples found."

            docs = vectorstore.similarity_search(query, k=k)
            if not docs:
                return "No specific examples found."

            examples = []
            for doc in docs:
                text = doc.page_content.strip()
                platform = doc.metadata.get("platform", "Unknown")
                toxicity = doc.metadata.get("toxicity", 0.0)
                polarized = "Yes" if doc.metadata.get("is_polarized", False) else "No"
                sentiment = doc.metadata.get("sentiment", "neutral")
                examples.append(
                    f"- Comment: \"{text}\" | Platform: {platform} | Toxicity: {toxicity:.3f} | Polarized: {polarized} | Sentiment: {sentiment}"
                )
            return "\n".join(examples)
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

        # Calculate exact metrics for each platform present in the current dataframe
        platform_stats = []
        for plat in sorted(filtered_df['platform'].unique()):
            plat_df = filtered_df[filtered_df['platform'] == plat]
            plat_total = len(plat_df)
            plat_pol = plat_df['is_polarized'].mean() * 100
            plat_tox = plat_df['toxicity'].mean()
            
            plat_pol_df = plat_df[plat_df['is_polarized'] == True]
            plat_non_pol_df = plat_df[plat_df['is_polarized'] == False]
            plat_tox_pol = plat_pol_df['toxicity'].mean() if not plat_pol_df.empty else 0.0
            plat_tox_non_pol = plat_non_pol_df['toxicity'].mean() if not plat_non_pol_df.empty else 0.0
            
            platform_stats.append(
                f"- Platform '{plat}':\n"
                f"  * Total messages: {plat_total}\n"
                f"  * Polarization rate: {plat_pol:.1f}%\n"
                f"  * Average toxicity: {plat_tox:.3f}\n"
                f"  * Average toxicity of polarized ('Us vs Them') messages: {plat_tox_pol:.3f}\n"
                f"  * Average toxicity of normal messages: {plat_tox_non_pol:.3f}"
            )
        platform_stats_str = "\n".join(platform_stats)

        context = (
            f"GLOBAL STATISTICS CONTEXT:\n"
            f"- Total analyzed messages: {total_msgs}\n"
            f"- Polarization Rate (Us vs Them language): {pol_pct:.1f}%\n"
            f"- Average Toxicity Score: {avg_tox:.3f}\n"
            f"- Average Toxicity for Normal messages: {avg_tox_non_pol:.3f}\n"
            f"- Average Toxicity for Polarized messages: {avg_tox_pol:.3f}\n\n"
            f"DETAILED STATISTICS BY PLATFORM:\n"
            f"{platform_stats_str}\n"
        )
        return context

    def generate_system_prompt(self, stats_context: str, retrieved_docs_text: str) -> str:
        """Assembles the system template using data and retrieved documents."""
        system_prompt = (
            "You are 'Project Shield AI', a professional data analyst assistant.\n"
            "You analyze social media polarization and toxicity. Answer concisely and naturally in a conversational way.\n"
            "You are an expert AI Agent. You must NEVER make up numbers or statistics.\n"
            "You must EXCLUSIVELY use the global and platform-specific statistics dynamically calculated from the real dataset below. Do not cite any figures that do not explicitly appear in this context.\n\n"
            "Use the following GLOBAL STATISTICS and RETRIEVED EXAMPLES to answer the user accurately.\n"
            "Never make up statistics. If the data provides the answer, use it.\n\n"
            f"{stats_context}\n\n"
            f"RETRIEVED EXAMPLES FROM DATASET:\n"
            f"{retrieved_docs_text}"
        )
        return system_prompt

    def get_extreme_comments_context(self, df: pd.DataFrame, top_n: int = 10) -> str:
        """Extracts the top toxic and top polarized comments directly from the pandas DataFrame."""
        if df.empty:
            return "No comments available in the current dataset."

        scored_df = df[df['is_scored'] == True]
        if scored_df.empty:
            return "No scored comments available."

        top_toxic = scored_df.sort_values(by='toxicity', ascending=False).head(top_n)
        
        toxic_lines = []
        for _, row in top_toxic.iterrows():
            text = str(row['clean_text']).strip()
            platform = row.get('platform', 'Unknown')
            toxicity = row.get('toxicity', 0.0)
            polarized = "Yes" if row.get('is_polarized', False) else "No"
            toxic_lines.append(
                f"- [Toxicity: {toxicity:.3f}] \"{text}\" (Platform: {platform}, Polarized: {polarized})"
            )

        polarized_df = df[df['is_polarized'] == True]
        top_polarized = polarized_df.sort_values(by='toxicity', ascending=False).head(top_n)
        
        pol_lines = []
        for _, row in top_polarized.iterrows():
            text = str(row['clean_text']).strip()
            platform = row.get('platform', 'Unknown')
            toxicity = row.get('toxicity', 0.0)
            pol_lines.append(
                f"- [Toxicity: {toxicity:.3f}] \"{text}\" (Platform: {platform}, Polarized: Yes)"
            )

        context_str = "EXTREME EXAMPLES FROM THE DATASET:\n\n"
        context_str += "Top 10 Most Toxic Comments:\n"
        context_str += "\n".join(toxic_lines) if toxic_lines else "None found."
        context_str += "\n\nTop 10 Most Toxic Polarized ('Us vs Them') Comments:\n"
        context_str += "\n".join(pol_lines) if pol_lines else "None found."
        
        return context_str

    def extract_matching_comments_for_query(self, query: str, df: pd.DataFrame, max_results: int = 10) -> str:
        """Parses the user query for toxicity thresholds (e.g. > 0.7, < 0.9) and extracts matching comments from the DataFrame."""
        if df.empty:
            return ""

        query_lower = query.lower()
        scored_df = df[df['is_scored'] == True]
        if scored_df.empty:
            return ""

        normalized_query = query_lower.replace(",", ".")
        import re
        numbers = [float(n) for n in re.findall(r'\b\d+(?:\.\d+)?\b', normalized_query)]
        
        greater_than = None
        less_than = None
        
        if len(numbers) == 1:
            val = numbers[0]
            if 0.0 <= val <= 1.0:
                if any(x in normalized_query for x in ["plus de", "supérieur", "superieur", ">", "au-dessus", "au dessus", "plus grand"]):
                    greater_than = val
                elif any(x in normalized_query for x in ["moins de", "inférieur", "inferieur", "<", "en-dessous", "en dessous", "plus petit"]):
                    less_than = val
                else:
                    greater_than = max(0.0, val - 0.1)
                    less_than = min(1.0, val + 0.1)
        elif len(numbers) >= 2:
            vals = sorted(numbers)
            if 0.0 <= vals[0] <= 1.0 and 0.0 <= vals[1] <= 1.0:
                greater_than = vals[0]
                less_than = vals[1]

        matching_df = scored_df
        if greater_than is not None:
            matching_df = matching_df[matching_df['toxicity'] >= greater_than]
        if less_than is not None:
            matching_df = matching_df[matching_df['toxicity'] <= less_than]

        if matching_df.equals(scored_df) or matching_df.empty:
            if "plus toxique" in normalized_query or "most toxic" in normalized_query:
                matching_df = scored_df.sort_values(by='toxicity', ascending=False).head(max_results)
            else:
                return ""

        matching_df = matching_df.sort_values(by='toxicity', ascending=False).head(max_results)
        lines = []
        for _, row in matching_df.iterrows():
            text = str(row['clean_text']).strip()
            platform = row.get('platform', 'Unknown')
            toxicity = row.get('toxicity', 0.0)
            polarized = "Yes" if row.get('is_polarized', False) else "No"
            lines.append(
                f"- [Toxicity: {toxicity:.3f}] \"{text}\" (Platform: {platform}, Polarized: {polarized})"
            )

        cond_desc = []
        if greater_than is not None:
            cond_desc.append(f"Toxicity >= {greater_than}")
        if less_than is not None:
            cond_desc.append(f"Toxicity <= {less_than}")
        cond_str = " and ".join(cond_desc) if cond_desc else "Most Toxic"

        context_str = f"\n\nCOMMENTS MATCHING QUERY CONDITION ({cond_str}):\n"
        context_str += "\n".join(lines)
        return context_str

    def ask(self, prompt: str, history: List[Dict[str, str]], filtered_df: pd.DataFrame) -> str:
        """Sends the question along with chat history and vector search context to the LLM."""
        from models.safety_guardrails import SafetyGuardrail
        guardrail = SafetyGuardrail()
        if guardrail.is_ultra_toxic(prompt):
            return "🛡️ **Safety Policy Warning**: Your prompt contains an ultra-toxic slur or variant. This request was blocked by Project Shield AI's guardrails."

        self._initialize_llm()
        assert self._chat_model is not None

        # Build Context
        stats_context = self.build_stats_context(filtered_df)
        retrieved_docs_text = self.retrieve_context_documents(prompt)
        
        # Add extreme examples and matching query conditions
        extreme_context = self.get_extreme_comments_context(filtered_df)
        matching_context = self.extract_matching_comments_for_query(prompt, filtered_df)
        
        additional_context = f"\n\n{extreme_context}{matching_context}"
        system_prompt = self.generate_system_prompt(stats_context, retrieved_docs_text + additional_context)

        # Assemble full message chain
        messages_for_llm = [SystemMessage(content=system_prompt)]

        for msg in history:
            if msg["role"] == "user":
                messages_for_llm.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages_for_llm.append(AIMessage(content=msg["content"]))

        response = self._chat_model.invoke(messages_for_llm)
        return response.content
