import requests
import pandas as pd
from datetime import datetime
import praw

# 配置参数
subreddit = "technology"  # 替换为你感兴趣的 subreddit
year = 2023
keyword = "chatgpt plus vs claude pro"  # 关键词
size = 100  # 每次请求返回的最大评论数量

# 构造时间范围
start_epoch = int(datetime(year, 10, 1).timestamp())
end_epoch = int(datetime(year + 1, 10, 1).timestamp())

# Reddit API 配置
client_id = 'bVWnv9TwGrAB5uvaPfHVKQ'
client_secret = 'czhCee56aaITDPrIEMUUNAxbxcmKTA'
user_agent = 'User-Agent: R script:Comparison-727:v1.0 (by /u/Substantial_Mode_278)'

# 使用 PRAW 库连接 Reddit API
reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent=user_agent)

# 搜索评论
comments = []
for submission in reddit.subreddit(subreddit).search(keyword, time_filter='year'):
    submission.comments.replace_more(limit=None)
    for comment in submission.comments.list():
        created_utc = datetime.utcfromtimestamp(comment.created_utc)
        if start_epoch <= comment.created_utc <= end_epoch:
            author_name = comment.author.name if comment.author else "[deleted]"
            comments.append({
                'author': author_name,
                'comment': comment.body,
                'created_utc': created_utc,
                'score': comment.score,
                'permalink': f"https://www.reddit.com{comment.permalink}"
            })

# 转为 DataFrame 并展示
df = pd.DataFrame(comments)
df.to_csv('reddit_comments.csv', index=False)

print("Comments have been saved to reddit_comments.csv")