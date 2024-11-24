import praw
import pandas as pd
from datetime import datetime

# Parameters
subreddits = ["ChatGPT", "tech", "ClaudeAI", "technology"]  # Subreddits to search
keywords = [
    "gpt plus",
    "claude pro",
    "gpt and claude",
    "chatgpt premium",
    "claude premium",
    "chatgpt plus",
    "chatgpt upgrade",
    "claude upgrade"
]  # Keywords to search for
max_posts_per_subreddit = 50  # Maximum posts per subreddit
max_comments_per_post = 40  # Maximum comments per post
output_file_posts = "reddit_praw_posts.csv"  # Output file for posts
output_file_comments = "reddit_praw_comments.csv"  # Output file for comments

# Reddit API credentials
client_id = "bVWnv9TwGrAB5uvaPfHVKQ"
client_secret = "czhCee56aaITDPrIEMUUNAxbxcmKTA"
user_agent = (
    "User-Agent: Python script:PRAW-Posts-Comments:v1.0 (by /u/YourRedditUsername)"
)

# Connect to Reddit API
reddit = praw.Reddit(
    client_id=client_id, client_secret=client_secret, user_agent=user_agent
)


# Function to fetch posts using PRAW
def get_posts_from_reddit(subreddit_name, keywords, max_posts):
    posts = []
    subreddit = reddit.subreddit(subreddit_name)
    post_count = 0

    for keyword in keywords:
        print(f"Searching for keyword '{keyword}' in r/{subreddit_name}...")
        for submission in subreddit.search(keyword, sort="new", time_filter="all"):
            if post_count >= max_posts:
                break
            posts.append(
                {
                    "id": submission.id,
                    "title": submission.title,
                    "created_utc": datetime.fromtimestamp(submission.created_utc),
                    "score": submission.score,
                    "author": submission.author.name
                    if submission.author
                    else "[deleted]",
                    "link": f"https://www.reddit.com{submission.permalink}",
                    "keyword": keyword,
                }
            )
            post_count += 1
            print(
                f"[{post_count}/{max_posts}] Fetched post: {submission.title[:50]}..."
            )
    return posts


# Function to fetch comments for a list of posts
def fetch_comments_for_posts(post_ids, max_comments):
    all_comments = []

    for post_id in post_ids:
        submission = reddit.submission(id=post_id)
        submission.comments.replace_more(limit=None)  # Expand all comments
        comment_count = 0
        for comment in submission.comments.list():
            if comment_count >= max_comments:
                break
            all_comments.append(
                {
                    "post_id": post_id,
                    "author": comment.author.name if comment.author else "[deleted]",
                    "comment": comment.body,
                    "created_utc": datetime.fromtimestamp(comment.created_utc),
                    "score": comment.score,
                    "permalink": f"https://www.reddit.com{comment.permalink}",
                }
            )
            comment_count += 1
    return all_comments


# Main script
all_posts = []
all_comments = []

for subreddit in subreddits:
    # Fetch posts using PRAW
    posts = get_posts_from_reddit(subreddit, keywords, max_posts_per_subreddit)
    all_posts.extend(posts)

    # Fetch comments for each post
    post_ids = [post["id"] for post in posts]
    comments = fetch_comments_for_posts(post_ids, max_comments_per_post)
    all_comments.extend(comments)

# Save posts and comments to CSV
posts_df = pd.DataFrame(all_posts)
comments_df = pd.DataFrame(all_comments)

posts_df.to_csv(output_file_posts, index=False)
comments_df.to_csv(output_file_comments, index=False)

print(f"Saved {len(all_posts)} posts to {output_file_posts}")
print(f"Saved {len(all_comments)} comments to {output_file_comments}")
