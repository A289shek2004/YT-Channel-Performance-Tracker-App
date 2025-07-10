import pandas as pd

# Load video data
df = pd.read_csv("video_sentiments.csv")

# Basic cleanup
df.dropna(subset=["Video Title", "Views", "Tags", "Publish Time", "Sentiment Score"], inplace=True)

# Feature Engineering
df["Title Length"] = df["Video Title"].apply(len)
df["Tag Count"] = df["Tags"].apply(lambda x: len(str(x).split("|")) if pd.notna(x) else 0)
df["Publish Hour"] = pd.to_datetime(df["Publish Time"], errors='coerce').dt.hour
df["Day of Week"] = pd.to_datetime(df["Published At"], errors='coerce').dt.day_name()

# Optional: Encode categorical
df = pd.get_dummies(df, columns=["Day of Week"], drop_first=True)

# Save for modeling
df[[
    "Views", "Title Length", "Tag Count", "Sentiment Score", "Publish Hour"
] + [col for col in df.columns if col.startswith("Day of Week_")]].to_csv("model_data.csv", index=False)

print("✅ Model data saved to model_data.csv")
