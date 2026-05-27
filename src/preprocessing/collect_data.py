import os
from typing import Any, Dict, List, Optional
import praw
import pandas as pd
from dotenv import load_dotenv


class RedditDataCollector:
    """Collector class for fetching social data from Reddit using PRAW."""

    def __init__(self) -> None:
        load_dotenv()
        self.client_id: Optional[str] = os.getenv("REDDIT_CLIENT_ID")
        self.client_secret: Optional[str] = os.getenv("REDDIT_CLIENT_SECRET")
        self.user_agent: Optional[str] = os.getenv("REDDIT_USER_AGENT")
        self._reddit: Optional[praw.Reddit] = None

    def _initialize_client(self) -> bool:
        """Initializes the Reddit API client if credentials are present."""
        if self._reddit is not None:
            return True

        if not all([self.client_id, self.client_secret, self.user_agent]):
            print("Warning: Missing Reddit credentials. Reddit collection will be skipped.")
            return False

        self._reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent
        )
        return True

    def fetch_comments_from_submission(self, submission: Any, comment_limit: int = 30) -> List[Dict[str, Any]]:
        """Extracts data for a set of comments from a single submission."""
        submission.comments.replace_more(limit=0)
        comments_data = []
        for comment in submission.comments.list()[:comment_limit]:
            comments_data.append({
                "comment_id": comment.id,
                "text": comment.body,
                "score": comment.score,
                "created_at": comment.created_utc,
            })
        return comments_data

    def fetch_subreddit_comments(
        self, 
        subreddit_name: str, 
        post_limit: int = 10, 
        comment_limit: int = 30
    ) -> List[Dict[str, Any]]:
        """Fetches comments from hot submissions of a specified subreddit."""
        if not self._initialize_client():
            return []

        assert self._reddit is not None
        subreddit = self._reddit.subreddit(subreddit_name)
        comments_list: List[Dict[str, Any]] = []

        print(f"Fetching data from r/{subreddit_name}...")
        for submission in subreddit.hot(limit=post_limit):
            sub_comments = self.fetch_comments_from_submission(submission, comment_limit)
            for comment_data in sub_comments:
                comment_data["subreddit"] = subreddit_name
                comments_list.append(comment_data)

        return comments_list

    def save_comments(self, comments: List[Dict[str, Any]], output_path: str) -> None:
        """Saves list of comments to a CSV file."""
        if not comments:
            print("No comments to save.")
            return

        df = pd.DataFrame(comments)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Success! {len(df)} comments saved to {output_path}")

    def collect_sport_data(
        self, 
        subreddit_name: str = "soccer", 
        post_limit: int = 10, 
        output_path: str = "data/raw/reddit_sport_raw.csv"
    ) -> None:
        """Runs the collection pipeline and saves results."""
        comments = self.fetch_subreddit_comments(subreddit_name, post_limit)
        self.save_comments(comments, output_path)


if __name__ == "__main__":
    collector = RedditDataCollector()
    collector.collect_sport_data()