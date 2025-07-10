import os
import pandas as pd
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import download
from datetime import datetime
from dotenv import load_dotenv

# Download VADER lexicon
download('vader_lexicon')

# Set your YouTube API key here
load_dotenv()
api_key = os.getenv("YOUTUBE_API_KEY")


# Channel IDs (replace/add as needed)
channel_ids = [
    "UCJZ7f6NQzGKZnFXzFW9y9UQ",  # Shaytards
    "UCYO_jab_esuFRV4b17AJtAw",  # 3Blue1Brown
    "UC8butISFwT-Wl7EV0hUK0BQ"   # freeCodeCamp
]

def get_channel_stats(channel_id):
    url = f"https://www.googleapis.com/youtube/v3/channels?part=snippet,statistics&id={channel_id}&key={api_key}"
    response = requests.get(url).json()

    if "items" not in response or not response["items"]:
        return {}

    data = response["items"][0]
    return {
        "Channel ID": channel_id,
        "Channel Name": data["snippet"]["title"],
        "Subscribers": int(data["statistics"].get("subscriberCount", 0)),
        "Total Views": int(data["statistics"].get("viewCount", 0)),
        "Total Videos": int(data["statistics"].get("videoCount", 0))
    }


subs_log_file = "subs_timeseries.csv"

for channel in channel_ids:
    stats = get_channel_stats(channel)
    today = datetime.date.today().isoformat()

    new_row = {
        "date": today,
        "channel_name": stats["Channel Name"],
        "subscribers": int(stats.get("subscriberCount", 0))
    }

    # Append to CSV
    if os.path.exists(subs_log_file):
        df_old = pd.read_csv(subs_log_file)
        df_new = pd.DataFrame([new_row])
        df = pd.concat([df_old, df_new], ignore_index=True)
        df.to_csv(subs_log_file, index=False)
    else:
        pd.DataFrame([new_row]).to_csv(subs_log_file, index=False)

def get_recent_videos(channel_id, max_results=5):
    url = f"https://www.googleapis.com/youtube/v3/search?key={api_key}&channelId={channel_id}&part=snippet,id&order=date&maxResults={max_results}"
    response = requests.get(url).json()

    videos = []
    for item in response.get("items", []):
        if item["id"]["kind"] == "youtube#video":
            video = {
                "Video ID": item["id"]["videoId"],
                "Video Title": item["snippet"]["title"],
                "Upload Date": item["snippet"]["publishedAt"][:10],
            }
            videos.append(video)
    return videos

def get_video_details(video_id):
    url = f"https://www.googleapis.com/youtube/v3/videos?part=statistics&id={video_id}&key={api_key}"
    response = requests.get(url).json()
    if "items" not in response or not response["items"]:
        return {}
    
    stats = response["items"][0]["statistics"]
    return {
        "Views": int(stats.get("viewCount", 0)),
        "Likes": int(stats.get("likeCount", 0)) if "likeCount" in stats else 0,
        "Comments": int(stats.get("commentCount", 0)) if "commentCount" in stats else 0
    }

def get_video_comments(video_id):
    comments = []
    url = f"https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={video_id}&key={api_key}&maxResults=100"
    response = requests.get(url).json()
    for item in response.get("items", []):
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(comment)
    return comments

def analyze_video_comments(comments):
    analyzer = SentimentIntensityAnalyzer()
    all_text = " ".join(comments)
    if not all_text.strip():
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0, "total_comments": 0}
    scores = analyzer.polarity_scores(all_text)
    scores["total_comments"] = len(comments)
    return scores

# Store channel and video stats
channel_stats_list = []
video_sentiment_list = []

print("Fetching data for", len(channel_ids), "channels...")

for channel_id in channel_ids:
    print("Processing Channel:", channel_id)
    channel_stats = get_channel_stats(channel_id)
    if not channel_stats:
        print("   Skipping channel (data not found)")
        continue

    channel_stats_list.append(channel_stats)

    recent_videos = get_recent_videos(channel_id, max_results=5)
    for video in recent_videos:
        print("  Analyzing:", video["Video Title"])
        video_id = video["Video ID"]

        # Fetch views
        video_details = get_video_details(video_id)
        video["Views"] = video_details.get("Views", 0)

        # Fetch & analyze comments
        comments = get_video_comments(video_id)
        sentiment = analyze_video_comments(comments)

 
# Inside your loop, where you're building the video_sentiment_list.append({...}):
upload_datetime = pd.to_datetime(video["Upload Date"])  # make sure Upload Date is in ISO format

video_sentiment_list.append({
    "Channel ID": channel_id,
    "Channel Name": channel_stats["Channel Name"],
    "Video Title": video["Video Title"],
    "Video ID": video["Video ID"],
    "Upload Date": video["Upload Date"],
    "Publish Time": upload_datetime.strftime("%H:%M"),
    "Publish Day": upload_datetime.day_name(),
    "Title Length": len(video["Video Title"]),
    "Views": video["Views"],
    "Sentiment Score": sentiment["compound"],
    "% Positive": round(sentiment["pos"] * 100, 2),
    "% Negative": round(sentiment["neg"] * 100, 2),
    "% Neutral": round(sentiment["neu"] * 100, 2),
    "Total Comments": sentiment["total_comments"],
    "Subscribers": int(channel_stats.get("subscriberCount", 0))  # Use get() to avoid KeyError
})

# Save results to CSV
channel_df = pd.DataFrame(channel_stats_list)
video_df = pd.DataFrame(video_sentiment_list)

channel_df.to_csv("channel_stats.csv", index=False)
video_df.to_csv("video_sentiments.csv", index=False)

print("\n Data saved to channel_stats.csv and video_sentiments.csv")
