from googleapiclient.discovery import build
import pandas as pd
from datetime import datetime
import json

# YouTube API Key
API_KEY = 'AIzaSyDl5CgbYYKYiAJqxfWoQxE6xQvBfL__4vY'
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)

def search_videos(keyword, published_after, max_results=5, pages=4):
    """Search for videos with pagination to ensure unique results."""
    processed_videos = load_processed_videos()

    # Pagination logic
    next_page_token = None
    for _ in range(pages):  # Loop through the number of pages
        response = youtube.search().list(
            q=keyword,
            part='id,snippet',
            maxResults=max_results,
            type='video',
            publishedAfter=published_after,
            pageToken=next_page_token
        ).execute()

        video_ids = [item['id']['videoId'] for item in response.get('items', [])]
        # Filter out already processed videos
        new_video_ids = [vid for vid in video_ids if vid not in processed_videos]

        # Update processed_videos and save
        processed_videos.update(new_video_ids)
        save_processed_videos(processed_videos)

        # Get nextPageToken for pagination
        next_page_token = response.get('nextPageToken')
        if not next_page_token:  # No more pages
            break

    return new_video_ids


def get_video_comments(video_id):
    """Extract all comments of the specified video"""
    comments = []
    response = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id,
        maxResults=100,
        textFormat='plainText'
    ).execute()

    while response:
        for item in response.get('items', []):
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            published_at = item['snippet']['topLevelComment']['snippet']['publishedAt']
            comments.append({
                'video_id': video_id,
                'comment': comment,
                'published_at': published_at
            })

        # Check if there is a next page
        if 'nextPageToken' in response:
            response = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                pageToken=response['nextPageToken'],
                maxResults=100,
                textFormat='plainText'
            ).execute()
        else:
            break

    return comments

def filter_comments_by_date(comments, start_date, end_date):
    """Filter comments by time range"""
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    filtered_comments = [
        comment for comment in comments
        if start_date <= datetime.strptime(comment['published_at'], "%Y-%m-%dT%H:%M:%SZ") <= end_date
    ]
    return filtered_comments

# Save collection to file
def save_processed_videos(processed_videos, filename="processed_videos.json"):
    with open(filename, 'w') as file:
        json.dump(list(processed_videos), file)

# Load collection from file
def load_processed_videos(filename="processed_videos.json"):
    try:
        with open(filename, 'r') as file:
            return set(json.load(file))
    except FileNotFoundError:
        return set()




def main(keyword, published_after, start_date, end_date, output_csv):

    video_ids = search_videos(keyword, published_after)
    all_comments = []

    for video_id in video_ids:
        print(f"Extracting reviews for video {video_id}...")
        comments = get_video_comments(video_id)
        filtered_comments = filter_comments_by_date(comments, start_date, end_date)
        all_comments.extend(filtered_comments)

    # Convert comments to DataFrame
    df = pd.DataFrame(all_comments)

    # Append new data to the existing CSV
    if not df.empty:
        try:
            # Check if file exists, if not write with header
            with open(output_csv, 'a', encoding='utf-8-sig') as f:
                # Append without header if the file exists
                df.to_csv(f, index=False, mode='a', header=f.tell() == 0)
        except FileNotFoundError:
            # If file doesn't exist, create a new one with header
            df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    else:
        print("No new comments found.")

    print(f"Comments have been successfully updated in {output_csv}")


if __name__ == "__main__":
    keyword = "ChatGPT Plus vs Claude Pro"
    published_after = "2023-10-01T00:00:00Z"
    start_date = "2023-10-01"
    end_date = "2024-10-01"
    output_csv = "youtube_comments.csv"

    main(keyword, published_after, start_date, end_date, output_csv)