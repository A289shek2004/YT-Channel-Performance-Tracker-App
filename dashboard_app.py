import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide", page_title="YouTube Channel Performance Tracker + Sentiment Analysis")
st.title("YouTube Channel Performance Tracker + Sentiment Analysis")
st.sidebar.markdown("##  Navigation")
if st.sidebar.button(" Subscriber Forecasting"):
    st.switch_page("pages/subscriber_forecasting.py")

# Load the data
try:
    video_stats = pd.read_csv("video_sentiments.csv")
except FileNotFoundError:
    st.error(" 'video_sentiments.csv' not found. Please run 'youtube_data_fetcher.py' first.")
    st.stop()

# Ensure required column exists
if "Channel Name" not in video_stats.columns:
    st.error(" 'Channel Name' column missing in CSV. Ensure correct export from fetcher script.")
    st.stop()

# Dropdown to select a channel
selected_channel = st.selectbox("🎥 Select a YouTube Channel", sorted(video_stats["Channel Name"].unique()))

# Filter data
video_stats = video_stats[video_stats["Channel Name"] == selected_channel].copy()
if "Published At" in video_stats.columns:
    video_stats["Published At"] = pd.to_datetime(video_stats["Published At"], errors='coerce')
    video_stats = video_stats.dropna(subset=["Published At"])
    video_stats["Week"] = video_stats["Published At"].dt.to_period("W").apply(lambda r: r.start_time)

    weekly_data = video_stats.groupby("Week").agg({
        "Video Title": "count",
        "Views": "sum"
    }).rename(columns={"Video Title": "Uploads"})

    weekly_data["Views Change %"] = weekly_data["Views"].pct_change() * 100
    weekly_data["Uploads Change %"] = weekly_data["Uploads"].pct_change() * 100
else:
    weekly_data = pd.DataFrame()
st.write("Available columns:", video_stats.columns.tolist())


# Display Channel Metrics
col1, col2, col3 = st.columns(3)

# Subscribers
subs = video_stats["Subscribers"].dropna().iloc[0] if "Subscribers" in video_stats.columns and not video_stats["Subscribers"].dropna().empty else None
try:
    subs = int(subs)
    subs_display = f"{subs:,}"
except:
    subs_display = "N/A"
col1.metric(" Subscribers", subs_display)

# Total Videos
col2.metric(" Total Videos", len(video_stats))

# Total Views
try:
    total_views = video_stats["Views"].astype(float).sum()
    views_display = f"{int(total_views):,}"
except:
    views_display = "N/A"
col3.metric(" Views", views_display)

# Display Weekly Growth Indicators
st.subheader(" Weekly Growth Indicators")
if not weekly_data.empty and len(weekly_data) >= 2:
    ...
else:
    st.info("Not enough weekly data to show growth indicators.")

# Check if data has enough points
if len(weekly_data) >= 2:
    latest = weekly_data.iloc[-1]
    prev = weekly_data.iloc[-2]

    col1, col2 = st.columns(2)
    with col1:
        delta_uploads = latest["Uploads"] - prev["Uploads"]
        st.metric("Uploads This Week", latest["Uploads"], f"{delta_uploads:+}")
    with col2:
        delta_views = latest["Views"] - prev["Views"]
        st.metric("Views This Week", f"{latest['Views']:,}", f"{delta_views:+,}")

    st.line_chart(weekly_data[["Views", "Uploads"]])
else:
    st.info("Not enough weekly data to show growth indicators.")

# Check for required columns
required_columns = ["Sentiment Score", "Views"]
missing_cols = [col for col in required_columns if col not in video_stats.columns]

if missing_cols:
    st.warning(f" Missing columns: {missing_cols}. Some analyses may be skipped.")
    combined_df = video_stats  # fallback
else:
    # Drop rows with missing sentiment or views
    combined_df = video_stats.dropna(subset=required_columns)

    # Correlation plot
    st.subheader(" Correlation between Sentiment & Views")
    correlation = combined_df["Sentiment Score"].corr(combined_df["Views"])
    if pd.notna(correlation):
        st.write(f"Correlation between sentiment and views: `{correlation:.2f}`")
    else:
        st.info("Not enough valid data to calculate correlation.")

    # Sentiment lift analysis
    st.subheader(" Key Insights")
    positive_videos = combined_df[combined_df["Sentiment Score"] > 0.3]
    neutral_or_negative = combined_df[combined_df["Sentiment Score"] <= 0.3]

    try:
        pos_avg = positive_videos["Views"].mean()
        base_avg = neutral_or_negative["Views"].mean()
        if pd.notna(pos_avg) and pd.notna(base_avg) and base_avg != 0:
            lift = ((pos_avg - base_avg) / base_avg) * 100
            st.metric("Videos with sentiment > 0.3 had", f"{lift:.2f}% more engagement")
        else:
            st.info("Not enough data to calculate sentiment-based engagement lift.")
    except:
        st.info("Could not calculate engagement lift due to missing data.")

# Sentiment Table
st.subheader("Recent Video Sentiment Analysis")
sentiment_cols = ["Video Title", "Sentiment Score", "% Positive", "% Neutral", "% Negative", "Views"]
if all(col in video_stats.columns for col in sentiment_cols):
    st.dataframe(video_stats[sentiment_cols].sort_values(by="Sentiment Score", ascending=False).head(10))
else:
    st.warning("Missing sentiment columns. Cannot display sentiment table.")
#  Title Length vs Views
if "Title Length" in video_stats.columns:
    st.subheader("📏 Title Length vs Views")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(data=video_stats, x="Title Length", y="Views", ax=ax1)
    ax1.set_yscale("log")  # Optional for better visibility
  # Optional for better visibility

# Publish Time vs Views
if "Publish Time" in video_stats.columns:
    st.subheader(" Publish Time vs Views")
    fig2, ax2 = plt.subplots()
    sns.boxplot(data=video_stats, x="Publish Time", y="Views", ax=ax2)
    ax2.set_title("Views by Publish Time (HH:MM)")
    ax2.tick_params(axis='x', rotation=45)
    st.pyplot(fig2)

#  Day of Week vs Views
if "Publish Day" in video_stats.columns:
    st.subheader(" Day of Week vs Views")
    fig3, ax3 = plt.subplots()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    sns.boxplot(data=video_stats, x="Publish Day", y="Views", order=order, ax=ax3)
    ax3.set_title("Views by Day of Week")
    st.pyplot(fig3)

# Scatter Plot
if all(col in combined_df.columns for col in ["Sentiment Score", "Views"]):
    st.subheader(" Sentiment vs Views Scatter Plot")
    fig, ax = plt.subplots()
    sns.scatterplot(data=combined_df, x="Sentiment Score", y="Views", ax=ax)
    ax.set_title("Sentiment Score vs Views")
    st.pyplot(fig)



st.header(" Comment Sentiment Explorer")

try:
    comment_df = pd.read_csv("comment_sentiments.csv")
except FileNotFoundError:
    st.warning("comment_sentiments.csv not found. Please run comment_sentiment_fetcher.py first.")
    st.stop()
    
# Filter comments for selected channel
filtered_comments = comment_df[
    (comment_df["Video Title"].isin(video_stats["Video Title"])) &
    (comment_df["Channel Name"] == selected_channel)
] if "Channel Name" in comment_df.columns else comment_df
# Sentiment Distribution
st.subheader("Sentiment Distribution on Comments")
sentiment_counts = filtered_comments["Sentiment"].value_counts()
fig, ax = plt.subplots()
ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%", startangle=90)
ax.axis("equal")
st.pyplot(fig)

# Show sample comments
st.subheader("Sample Comments by Sentiment")

sentiment_choice = st.selectbox("Select Sentiment", ["Positive", "Neutral", "Negative"])
filtered = filtered_comments[filtered_comments["Sentiment"] == sentiment_choice]
st.dataframe(filtered[["Video Title", "Comment", "Sentiment"]].head(10))

# Optional: Most Positive/Negative Comments
st.subheader(" Top Positive & Negative Comments")

col1, col2 = st.columns(2)

with col1:
    st.markdown("** Most Positive Comments**")
    st.dataframe(
        filtered_comments.sort_values(by="Positive", ascending=False)[
            ["Video Title", "Comment", "Positive"]
        ].head(5)
    )

with col2:
    st.markdown("** Most Negative Comments**")
    st.dataframe(
        filtered_comments.sort_values(by="Negative", ascending=False)[
            ["Video Title", "Comment", "Negative"]
        ].head(5)
    )
# ----------------------------------------------
# ?Engagement Prediction Section
# ----------------------------------------------
st.header(" Predict Video Engagement (Views)")

try:
    model = joblib.load("engagement_model.pkl")
except FileNotFoundError:
    st.warning("Prediction model not found. Please run `engagement_model.py` to train it.")
else:
    required_model_features = [
        "Title Length", "Tag Count", "Sentiment Score", "Publish Hour"
    ] + [col for col in video_stats.columns if col.startswith("Day of Week_")]

    # Check if all model features exist
    if all(col in video_stats.columns for col in required_model_features):
        model_input = video_stats[required_model_features].dropna()
        predicted_views = model.predict(model_input)
        video_stats["Predicted Views"] = predicted_views

        st.success("Engagement prediction complete.")
        st.dataframe(video_stats[["Video Title", "Views", "Predicted Views"]].sort_values(by="Predicted Views", ascending=False).head(10))
    else:
        st.warning("Missing required columns for prediction. Please ensure feature engineering is done.")
