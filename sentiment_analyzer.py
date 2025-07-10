# sentiment_analyzer.py

from googleapiclient.discovery import build
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import download
import numpy as np
from dotenv import load_dotenv

download('vader_lexicon')

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")  
youtube = build("youtube", "v3", developerKey=API_KEY)

# Fetch comments for a given video ID
def get_comments(video_id, max_results=100):
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=min(max_results, 100),
        textFormat="plainText"
    )
    response = request.execute()

    for item in response.get("items", []):
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(comment)

    return comments

# Analyze sentiment
def analyze_video_comments(video_id):
    comments = get_comments(video_id)
    sid = SentimentIntensityAnalyzer()

    if not comments:
        return {
            "compound": 0.0,
            "pos": 0.0,
            "neg": 0.0,
            "neu": 0.0,
            "total_comments": 0
        }

    compound_scores = []
    pos_scores = []
    neg_scores = []
    neu_scores = []

    for comment in comments:
        score = sid.polarity_scores(comment)
        compound_scores.append(score["compound"])
        pos_scores.append(score["pos"])
        neg_scores.append(score["neg"])
        neu_scores.append(score["neu"])

    return {
        "compound": np.mean(compound_scores),
        "pos": np.mean(pos_scores),
        "neg": np.mean(neg_scores),
        "neu": np.mean(neu_scores),
        "total_comments": len(comments)
    }
