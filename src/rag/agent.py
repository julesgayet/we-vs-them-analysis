import os
import re
from typing import Any, Callable, Dict, List, Optional
import pandas as pd
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# Helper function to load dataset
def load_all_scored_data(scored_dir: str = "data/scored") -> pd.DataFrame:
    """Helper to load all scored comments from the scored folder, filtered by selected source in UI."""
    if not os.path.exists(scored_dir):
        return pd.DataFrame()
    dfs = []
    for root, _, files in os.walk(scored_dir):
        for file in files:
            if not file.endswith(".csv"):
                continue
            try:
                temp_df = pd.read_csv(os.path.join(root, file), low_memory=False)
                # Assign platform
                rel_path = os.path.relpath(os.path.join(root, file), scored_dir)
                path_parts = os.path.normpath(rel_path).split(os.sep)
                platform = path_parts[0].capitalize() if len(path_parts) > 1 else "General"
                temp_df['platform'] = platform
                dfs.append(temp_df)
            except Exception:
                pass
    if not dfs:
        return pd.DataFrame()
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Filter based on active dataset source in streamlit session state if available
    try:
        import streamlit as st
        if "selected_source" in st.session_state:
            source = st.session_state.selected_source
            if source == "Dataset (Twitter, TikTok, Instagram)":
                df = df[df['platform'].isin(['Twitter', 'Tiktok', 'Instagram'])]
            elif source == "Hugging Face Dataset (Reddit)":
                df = df[df['platform'].isin(['Reddit'])]
            else:
                import re
                match = re.search(r'\((.*?)\)', source)
                if match:
                    df = df[df['platform'] == match.group(1)]
    except Exception:
        pass
        
    return df


# Tool Definitions
def get_platform_metrics(platform: str) -> str:
    """Analytical tool that pulls polarization and toxicity statistics for a platform."""
    df = load_all_scored_data()
    if df.empty:
        return "Error: No scored data found."

    plat = str(platform).strip().lower().replace("'", "").replace('"', '')
    plat = re.sub(r'^(platform\s*=\s*|platform\s*:\s*)', '', plat)
    df['platform_lower'] = df['platform'].str.lower()
    plat_df = df[df['platform_lower'] == plat]
    if plat_df.empty:
        return f"No metrics found for platform '{plat}'."

    total = len(plat_df)
    pol_rate = plat_df['is_polarized'].mean() * 100
    avg_tox = plat_df['toxicity'].mean()
    sentiment_dist = plat_df['sentiment'].value_counts(normalize=True).to_dict()
    sent_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in sentiment_dist.items()])

    return (
        f"Platform: {platform.capitalize()}\n"
        f"- Total Messages: {total}\n"
        f"- Polarization Rate: {pol_rate:.2f}%\n"
        f"- Average Toxicity Score: {avg_tox:.4f}\n"
        f"- Sentiment Distribution: {sent_str}"
    )


def get_spike_events() -> str:
    """Analytical tool that retrieves detected activity spikes and their associated sports matches."""
    path = "data/processed/detected_spikes.csv"
    if not os.path.exists(path):
        return "No spike event data found. Please run causal analysis first."

    try:
        df = pd.read_csv(path)
        if df.empty:
            return "No significant spikes detected."
        # Keep spikes with actual matches
        event_spikes = df[df['event'] != "Unknown Sports Event"]
        if event_spikes.empty:
            event_spikes = df.head(10)

        lines = []
        for _, row in event_spikes.iterrows():
            lines.append(
                f"- Date: {row['parsed_date']} | Comments Count: {row['comment_count']} | Event: {row['event']}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Error loading spikes: {e}"


def get_top_polarized_topics() -> str:
    """Analytical tool that extracts the most common words from polarized content."""
    df = load_all_scored_data()
    if df.empty:
        return "Error: No scored data found."

    polarized_df = df[df['is_polarized'] == True]
    if polarized_df.empty:
        return "No polarized content found to extract topics from."

    stop_words = {
        "this", "that", "with", "from", "the", "and", "you", "they", "them", 
        "have", "what", "their", "will", "your", "about", "there", "would"
    }
    words = " ".join(polarized_df['clean_text'].astype(str)).lower().split()
    filtered_words = [w for w in words if len(w) > 4 and w not in stop_words]

    from collections import Counter
    common = Counter(filtered_words).most_common(15)
    return ", ".join([f"{w} ({c})" for w, c in common])


def retrieve_similar_comments(query: str) -> str:
    """Retrieves real user comments from the dataset that match or are semantically close to a search query."""
    from rag.assistant import AIAssistant
    import streamlit as st
    allowed_platforms = None
    try:
        if "selected_source" in st.session_state:
            source = st.session_state.selected_source
            if source == "Dataset (Twitter, TikTok, Instagram)":
                allowed_platforms = ['Twitter', 'Tiktok', 'Instagram']
            elif source == "Hugging Face Dataset (Reddit)":
                allowed_platforms = ['Reddit']
            else:
                import re
                match = re.search(r'\((.*?)\)', source)
                if match:
                    allowed_platforms = [match.group(1)]
    except Exception:
        pass
    assistant = AIAssistant()
    return assistant.retrieve_context_documents(query, k=5, allowed_platforms=allowed_platforms)


class FunctionCallingAgent:
    """Implements a ReAct (Reasoning and Action) loop to answer NL queries by executing python analytical tools."""

    def __init__(self) -> None:
        load_dotenv()
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if hf_token:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

        self._chat_model: Optional[ChatHuggingFace] = None
        self.tools: Dict[str, Callable[..., str]] = {
            "get_platform_metrics": get_platform_metrics,
            "get_spike_events": get_spike_events,
            "get_top_polarized_topics": get_top_polarized_topics,
            "retrieve_similar_comments": retrieve_similar_comments
        }


    def _initialize_llm(self) -> None:
        """Loads Qwen endpoint wrapped in ChatHuggingFace for conversational support."""
        if self._chat_model is not None:
            return

        llm = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-7B-Instruct",
            task="text-generation",
            max_new_tokens=512,
            temperature=0.1,
            repetition_penalty=1.1
        )
        self._chat_model = ChatHuggingFace(llm=llm)

    def run(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None, dataset_context: Optional[str] = None, max_iterations: int = 5) -> str:
        """Runs the ReAct loop until a final answer is retrieved or iteration limit is reached."""
        from models.safety_guardrails import SafetyGuardrail
        guardrail = SafetyGuardrail()
        if guardrail.is_ultra_toxic(query):
            return "🛡️ **Safety Policy Warning**: Your query contains an ultra-toxic slur or variant. This request was blocked by the guardrails."

        self._initialize_llm()
        assert self._chat_model is not None

        system_prompt = (
            "You are the ReAct Data Analyst Agent. You solve questions step-by-step by thinking and invoking tools.\n"
            "You have access to the following tools:\n\n"
            "- get_platform_metrics(platform: str): Returns polarization rates, average toxicity, and sentiment percentages for a given platform name ('twitter', 'tiktok', 'instagram').\n"
            "- get_spike_events(): Returns a list of detected temporal activity spikes mapped to external soccer matches/events.\n"
            "- get_top_polarized_topics(): Returns the most common words found in polarized messages.\n"
            "- retrieve_similar_comments(query: str): Search and return real user comments from the dataset that match or are semantically close to a query topic.\n\n"
            "You MUST write all intermediate thoughts and actions in the EXACT format below:\n"
            "Thought: <your thought process>\n"
            "Action: <tool_name>(<argument>)\n"
            "(Wait for user Observation...)\n\n"
            "Once you have gathered the tool observations, write:\n"
            "Thought: I have the final answer\n"
            "Final Answer: <your final complete answer to the user>\n\n"
            "CRITICAL Rules:\n"
            "1. Action must be written EXACTLY as: Action: tool_name(argument). Do not put quotes around string arguments inside the parentheses.\n"
            "2. If you do not need to call a tool, directly write your Final Answer.\n"
        )

        if dataset_context:
            system_prompt += f"\n\nCURRENT ACTIVE DATASET CONTEXT:\n{dataset_context}\n"

        messages = [
            SystemMessage(content=system_prompt)
        ]

        if chat_history:
            # Filter out current user query from history to avoid duplicates
            history_to_process = chat_history
            if history_to_process and history_to_process[-1].get("role") == "user" and history_to_process[-1].get("content") == query:
                history_to_process = history_to_process[:-1]

            # Keep last 10 messages (approx 5 turns) for token efficiency
            for msg in history_to_process[-10:]:
                role = msg.get("role")
                content = msg.get("content", "")
                if not content:
                    continue
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))

        messages.append(HumanMessage(content=f"User Query: {query}"))
        
        iteration = 0


        while iteration < max_iterations:
            print(f"Agent Loop iteration {iteration + 1}...")
            response = self._chat_model.invoke(messages).content.strip()
            print(f"LLM Response:\n{response}\n")

            # Check if there is an Action
            action_match = re.search(r"Action:\s*([a-zA-Z0-9_]+)\((.*)\)", response)
            if action_match:
                tool_name = action_match.group(1).strip()
                tool_arg = action_match.group(2).strip()

                if tool_name in self.tools:
                    print(f"Executing Tool: {tool_name} with arg: {tool_arg}")
                    try:
                        if tool_arg:
                            observation = self.tools[tool_name](tool_arg)
                        else:
                            observation = self.tools[tool_name]()
                    except Exception as e:
                        observation = f"Error executing tool: {e}"
                else:
                    observation = f"Error: Tool '{tool_name}' does not exist."

                print(f"Tool Observation:\n{observation}\n")

                # Update messages history:
                messages.append(AIMessage(content=response))
                messages.append(HumanMessage(content=f"Observation: {observation}"))
                iteration += 1
            else:
                # No action found, look for Final Answer
                final_match = re.search(r"Final Answer:\s*(.*)", response, re.DOTALL)
                if final_match:
                    return final_match.group(1).strip()
                
                # Fallback if Final Answer keyword is missing but it's the final output
                if "Thought:" in response:
                    return response.split("Thought:")[-1].strip()
                return response

        return "Error: Agent reached maximum iterations without finding a final answer."


if __name__ == "__main__":
    agent = FunctionCallingAgent()
    test_query = "Show me the top topics with highest polarization and also platform metrics for Twitter."
    print("Running Test Query...")
    result = agent.run(test_query)
    print("\n--- Final Agent Output ---")
    print(result)
