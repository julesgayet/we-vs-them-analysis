import os
import praw
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_reddit_instance():
    """
    Initializes the Reddit API instance using environment variables
    """
    return praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT")
    )

def collect_sport_data(subreddit_name="soccer", post_limit=10):
    """
    Collects comments from a specific subreddit for 'Us vs Them' analysis.
    Focus: SPORT
    """
    reddit = get_reddit_instance()
    subreddit = reddit.subreddit(subreddit_name)
    
    comments_list = []
    
    print(f"Fetching data from r/{subreddit_name}...")
    
    for submission in subreddit.hot(limit=post_limit):
        submission.comments.replace_more(limit=0)
        for comment in submission.comments.list()[:30]:
            comments_list.append({
                "comment_id": comment.id,
                "text": comment.body,
                "score": comment.score,
                "created_at": comment.created_utc,
                "subreddit": subreddit_name
            })
            
    # Save to raw data folder
    df = pd.DataFrame(comments_list)
    output_path = "data/raw/reddit_sport_raw.csv"
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Success! {len(df)} comments saved to {output_path}")

if __name__ == "__main__":
    collect_sport_data()