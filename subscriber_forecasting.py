import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import streamlit as st

# Load the data
df = pd.read_csv("subs_timeseries.csv")

# Let user select a channel
st.title("📈 YouTube Subscriber Forecasting")
channels = df["channel_name"].unique()
selected = st.selectbox("Select a Channel", sorted(channels))

# Prepare data
df = df[df["channel_name"] == selected][["date", "subscribers"]].dropna()
df = df.rename(columns={"date": "ds", "subscribers": "y"})

# Fit model
model = Prophet()
model.fit(df)

# Forecast next 30 days
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Plot
st.subheader("Forecast Plot")
fig1 = model.plot(forecast)
st.pyplot(fig1)

# Optional: Show forecast data
st.subheader("Forecast Data (next 30 days)")
st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(30))
