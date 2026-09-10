import praw
import csv
import datetime
import os
import sys
from dotenv import load_dotenv

# Add root directory to sys.path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import DATA_RAW

# Load environment variables from .env
load_dotenv()

# --- CONFIGURATION ---
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = "Thesis_Data_Scraper_v1.0"

# List of subreddits to monitor
SUBREDDITS = ["ukraine", "worldnews", "RussiaUkraineWar2022", "UkraineWarVideoReport"]

# Time range for collection (e.g., from February 2022)
START_DATE = datetime.datetime(2025, 5, 25, tzinfo=datetime.timezone.utc)
LIMIT_POSTS = 500  # Number of posts to retrieve per subreddit

OUTPUT_FILE_REDDIT = os.path.join(DATA_RAW, "reddit_comments_raccolti.csv")

# --- DATA COLLECTION FUNCTION ---

def collect_reddit_data():
    """
    Initializes the Reddit client and retrieves top posts and comments 
    containing specific keywords related to the conflict.
    """
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )

    print("Connected to Reddit API.")
    
    keywords = ["ukraine", "russia", "war", "putin", "zelensky", "invasion"]

    try:
        with open(OUTPUT_FILE_REDDIT, 'w', newline='', encoding='utf-8') as file_csv:
            writer = csv.writer(file_csv)
            writer.writerow(["id_originale", "subreddit", "author", "text", "timestamp_utc", "score", "type"])

            for sub_name in SUBREDDITS:
                print(f"\n--- Scraping subreddit: r/{sub_name} ---")
                subreddit = reddit.subreddit(sub_name)
                
                # Fetch hot posts
                for submission in subreddit.hot(limit=LIMIT_POSTS):
                    submission_date = datetime.datetime.fromtimestamp(submission.created_utc, tz=datetime.timezone.utc)
                    
                    if submission_date < START_DATE:
                        continue
                    
                    # Check if title or text matches keywords
                    if any(kw in submission.title.lower() for kw in keywords):
                        # Save the post itself
                        writer.writerow([
                            submission.id, sub_name, str(submission.author), 
                            submission.title + " " + submission.selftext, 
                            submission.created_utc, submission.score, "post"
                        ])

                        # Fetch top comments
                        submission.comments.replace_more(limit=0)
                        for comment in submission.comments.list()[:20]: # Limit to top 20 comments per post
                            writer.writerow([
                                comment.id, sub_name, str(comment.author), 
                                comment.body, comment.created_utc, comment.score, "comment"
                            ])
                
                print(f"  Finished r/{sub_name}")
                
    except Exception as e:
        print(f"ERROR during Reddit scraping: {e}")

if __name__ == "__main__":
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        print("ERROR: Please provide REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in the .env file.")
    else:
        collect_reddit_data()
        print(f"\nReddit data saved to: {OUTPUT_FILE_REDDIT}")
