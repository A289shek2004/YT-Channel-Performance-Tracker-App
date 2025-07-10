# comment_sentiment_fetcher.py

from googleapiclient.discovery import build
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import download
import pandas as pd
import time
import os
from dotenv import load_dotenv

# Load API Key
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

youtube = build("youtube", "v3", developerKey=API_KEY)

download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def get_comments(video_id, max_comments=100):
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText",
            pageToken=next_page_token
        ).execute()

        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
            if len(comments) >= max_comments:
                break

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
        time.sleep(0.2)

    return comments

def analyze_comments(video_id, title):
    comments = get_comments(video_id)
    results = []

    for comment in comments:
        sentiment = sia.polarity_scores(comment)
        results.append({
            "Video ID": video_id,
            "Video Title": title,
            "Comment": comment,
            "Sentiment": sentiment['compound'],
            "Pos": sentiment['pos'],
            "Neu": sentiment['neu'],
            "Neg": sentiment['neg']
        })

    return results

# Example: run for a few videos
video_data = pd.read_csv("video_sentiments.csv")
all_comments = []

for idx, row in video_data.iterrows():
    video_id = row["Video ID"]
    title = row["Video Title"]
    print(f"Processing: {title}")
    comments_data = analyze_comments(video_id, title)
    all_comments.extend(comments_data)

df_comments = pd.DataFrame(all_comments)
df_comments.to_csv("comment_sentiments.csv", index=False)
print(" Comment sentiment analysis complete!")
