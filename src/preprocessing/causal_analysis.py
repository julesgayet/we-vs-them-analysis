import os
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import pandas as pd


class CausalAnalyzer:
    """Detects temporal spikes in sports comment volumes and maps them to external matches/events."""

    def __init__(self, scored_dir: str = "data/scored", output_dir: str = "data/processed") -> None:
        self.scored_dir = scored_dir
        self.output_dir = output_dir
        # Map of historical dates to key sports events/matches
        self.event_calendar = {
            "2025-04-15": "Champions League Quarter-Final (PSG vs Barcelona / Dortmund vs Atletico)",
            "2025-04-16": "Champions League Quarter-Final (Man City vs Real Madrid / Bayern vs Arsenal)",
            "2025-04-14": "Premier League Matchday / UCL Match Eve Anticipation",
            "2023-01-16": "Supercopa de España Final (Real Madrid vs Barcelona)",
            "2023-01-19": "Riyadh Season Cup (PSG vs Riyadh XI - Messi vs Ronaldo)",
            "2025-02-25": "Champions League Round of 16 Matches",
            "2025-02-14": "Valentine's Day / European League Fixtures",
            "2025-02-10": "Premier League Monday Night Football",
            "2022-04-28": "Europa League Semi-Finals First Leg",
            "2025-03-04": "Champions League Round of 16 Second Leg"
        }

    def load_scored_data(self) -> pd.DataFrame:
        """Loads and combines all scored CSV datasets."""
        if not os.path.exists(self.scored_dir):
            print(f"Directory not found: {self.scored_dir}")
            return pd.DataFrame()

        dfs: List[pd.DataFrame] = []
        for root, _, files in os.walk(self.scored_dir):
            for file in files:
                if not file.endswith(".csv"):
                    continue
                self._load_single_file(os.path.join(root, file), dfs)

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)

    def _load_single_file(self, filepath: str, dfs: List[pd.DataFrame]) -> None:
        """Helper to load a single scored CSV file and parse date."""
        try:
            df = pd.read_csv(filepath, low_memory=False)
            col = 'timestamp' if 'timestamp' in df.columns else ('Timestamp' if 'Timestamp' in df.columns else 'createTimeISO')
            if col not in df.columns:
                return

            df['parsed_date'] = pd.to_datetime(df[col], errors='coerce').dt.date
            df = df.dropna(subset=['parsed_date'])
            
            # Determine platform name
            rel_path = os.path.relpath(filepath, self.scored_dir)
            path_parts = os.path.normpath(rel_path).split(os.sep)
            platform = path_parts[0].capitalize() if len(path_parts) > 1 else "General"
            df['platform'] = platform
            
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    def detect_volume_spikes(self, df: pd.DataFrame, threshold_std: float = 1.5) -> pd.DataFrame:
        """Detects dates where comment volume is significantly above rolling average."""
        if df.empty:
            return pd.DataFrame()

        daily_counts = df.groupby('parsed_date').size().rename('comment_count').reset_index()
        daily_counts = daily_counts.sort_values(by='parsed_date').reset_index(drop=True)

        if len(daily_counts) < 7:
            mean_val = daily_counts['comment_count'].mean()
            std_val = daily_counts['comment_count'].std() if daily_counts['comment_count'].std() > 0 else 1.0
        else:
            rolling = daily_counts['comment_count'].rolling(window=7, min_periods=1)
            mean_val = rolling.mean()
            std_val = rolling.std().fillna(1.0)

        daily_counts['is_spike'] = daily_counts['comment_count'] > (mean_val + threshold_std * std_val)
        return daily_counts[daily_counts['is_spike'] == True]

    def map_spikes_to_events(self, spikes_df: pd.DataFrame, full_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Maps detected spikes to known external sport events, dynamically detecting them if needed."""
        if spikes_df.empty:
            return pd.DataFrame()

        if full_df is None:
            full_df = self.load_scored_data()

        events = []
        for _, row in spikes_df.iterrows():
            date_str = str(row['parsed_date'])
            # Check calendar first
            if date_str in self.event_calendar:
                events.append(self.event_calendar[date_str])
            else:
                # Try to dynamically detect event from comments on that date
                detected = self._detect_event_from_comments(date_str, full_df)
                events.append(detected)

        spikes_df['event'] = events
        return spikes_df

    def _detect_event_from_comments(self, date_str: str, full_df: pd.DataFrame) -> str:
        """Detects the event name from the comments on a given date using LLM or fallback frequency analysis."""
        date_comments = full_df[full_df['parsed_date'].astype(str) == date_str]
        if date_comments.empty:
            return "Unknown Event"

        comments = date_comments['clean_text'].dropna().head(30).tolist()
        if not comments:
            return "Unknown Event"

        # Try using Hugging Face LLM if token is present
        from dotenv import load_dotenv
        load_dotenv()
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
        if hf_token:
            try:
                os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
                from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
                from langchain_core.messages import SystemMessage, HumanMessage
                
                llm = HuggingFaceEndpoint(
                    repo_id="Qwen/Qwen2.5-7B-Instruct",
                    task="text-generation",
                    max_new_tokens=40,
                    temperature=0.1
                )
                chat = ChatHuggingFace(llm=llm)
                
                comments_str = "\n".join([f"- {c}" for c in comments])
                messages = [
                    SystemMessage(content=(
                        "You analyze social media comments from a specific date to identify the single main event, news, or match being discussed. "
                        "Respond with ONLY the name of the main event or topic (maximum 6 words), no extra text."
                    )),
                    HumanMessage(content=f"Date: {date_str}\nComments:\n{comments_str}")
                ]
                res = chat.invoke(messages).content.strip()
                if res:
                    return res
            except Exception as e:
                print(f"LLM event detection failed for {date_str}: {e}. Using fallback.")

        # Fallback keyword extraction
        return self._fallback_detect_event(date_str, comments)

    def _fallback_detect_event(self, date_str: str, comments: List[str]) -> str:
        """Fallback method to extract event name using word frequencies when LLM is unavailable."""
        if not comments:
            return "Unknown Event"

        text = " ".join(comments).lower()
        import re
        words = re.findall(r'\b[a-z]{4,15}\b', text)
        
        stop_words = {
            "this", "that", "with", "from", "they", "them", "have", "what", "their", 
            "will", "your", "about", "there", "would", "like", "just", "some", "more",
            "people", "about", "where", "when", "here", "there", "their", "being", "been",
            "were", "than", "then", "into", "could"
        }
        filtered = [w for w in words if w not in stop_words]
        
        if not filtered:
            return "Activity Spike"
            
        from collections import Counter
        common = Counter(filtered).most_common(3)
        keywords = ", ".join([w[0].capitalize() for w in common])
        
        sports_keywords = {"vs", "game", "match", "cup", "league", "win", "loss", "fc", "real", "barca", "madrid", "chelsea", "united", "psg", "liverpool", "bayern", "juventus", "milan", "inter", "city", "arsenal", "ucl", "champions"}
        has_sports = any(w in sports_keywords for w in filtered)
        
        if has_sports:
            return f"Sports Discussion ({keywords})"
        return f"Discussion Spike ({keywords})"

    def generate_causal_graph(self, output_path: str = "data/processed/causal_graph.png") -> None:
        """Generates and saves a beautiful causal flowchart using matplotlib."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')

        # Draw nodes and directed causal arrows
        node_style = dict(boxstyle="round,pad=0.5", fc="lightblue", ec="blue", lw=2)
        arrow_style = dict(arrowstyle="->", lw=2.5, color="gray")

        ax.text(0.1, 0.8, "External Match / Event\n(e.g., Champions League)", ha="center", va="center", bbox=node_style, fontsize=12)
        ax.text(0.5, 0.8, "Spike in Activity\n(High Volume of Comments)", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange", lw=2), fontsize=12)
        ax.text(0.9, 0.8, "Increased Toxicity\n& Out-group Hostility", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", fc="lightpink", ec="red", lw=2), fontsize=12)

        ax.text(0.5, 0.4, "We vs Them Polarization\n(In-group Bias)", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", fc="lightgreen", ec="green", lw=2), fontsize=12)

        # Arrows
        ax.annotate("", xy=(0.35, 0.8), xytext=(0.25, 0.8), arrowprops=arrow_style)
        ax.annotate("", xy=(0.75, 0.8), xytext=(0.65, 0.8), arrowprops=arrow_style)
        ax.annotate("", xy=(0.5, 0.52), xytext=(0.5, 0.68), arrowprops=arrow_style)
        ax.annotate("", xy=(0.8, 0.7), xytext=(0.6, 0.5), arrowprops=arrow_style)

        # Labels on arrows
        ax.text(0.3, 0.83, "triggers", ha="center", va="center", fontsize=10, fontstyle="italic")
        ax.text(0.7, 0.83, "exacerbates", ha="center", va="center", fontsize=10, fontstyle="italic")
        ax.text(0.52, 0.6, "drives out-grouping", ha="left", va="center", fontsize=10, fontstyle="italic")
        ax.text(0.72, 0.58, "correlates with", ha="left", va="center", fontsize=10, fontstyle="italic")

        ax.set_title("Causal Pathway of 'We vs Them' Polarization during Sports Events", fontsize=14, fontweight="bold", pad=20)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"✅ Saved causal graph visualization to {output_path}")

    def _is_football_related(self, event_name: str) -> bool:
        """Checks if the event description is related to football (soccer)."""
        event_lower = event_name.lower()
        
        football_keywords = [
            "champions league", "europa league", "premier league", "laliga", "serie a", "ligue 1",
            "ucl", "uel", "psg", "barcelona", "barca", "real madrid", "madrid", "bayern", 
            "dortmund", "atletico", "man city", "chelsea", "liverpool", "arsenal", "juventus", 
            "milan", "inter", "roma", "tottenham", "spurs", "manchester", "united",
            "messi", "ronaldo", "mbappe", "neymar", "haaland", "lamine", "yamal",
            "supercopa", "copa", "football", "soccer", "foot", "matchday", "el clasico", "clasico",
            "riyadh season cup", "leicester", "leeds", "everton", "newcastle",
            "fixture", "fixtures", "derby", "cup", "tournament"
        ]
        
        return any(kw in event_lower for kw in football_keywords)

    def run_analysis(self) -> None:
        """Runs the entire causal analysis pipeline."""
        df = self.load_scored_data()
        if df.empty:
            print("Error: No scored data found. Run scoring pipeline first.")
            return

        spikes = self.detect_volume_spikes(df)
        spikes_with_events = self.map_spikes_to_events(spikes, df)

        # Filter to keep only football-related events
        if not spikes_with_events.empty:
            spikes_with_events = spikes_with_events[
                spikes_with_events['event'].apply(self._is_football_related)
            ]

        print("\n--- Detected Activity Spikes and External Events ---")
        if spikes_with_events.empty:
            print("No significant activity spikes found.")
        else:
            print(spikes_with_events.to_string(index=False))

        # Save spikes list
        os.makedirs(self.output_dir, exist_ok=True)
        spikes_with_events.to_csv(os.path.join(self.output_dir, "detected_spikes.csv"), index=False)
        self.generate_causal_graph()


if __name__ == "__main__":
    analyzer = CausalAnalyzer()
    analyzer.run_analysis()
