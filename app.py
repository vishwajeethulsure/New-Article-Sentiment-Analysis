import streamlit as st
import pandas as pd
import os
from gtts import gTTS
from io import BytesIO

# Sample company list (You can expand this)
COMPANIES = ["Tesla", "Apple", "Microsoft", "Amazon", "Google"]

# Streamlit UI
st.title("📰 News Sentiment Analyzer")
st.subheader("🔍 Select a company to analyze its latest news sentiment")

# Dropdown for company selection
company_name = st.selectbox("Choose a company", COMPANIES)

# Load the news data (Replace this with actual scraping logic)
news_df = pd.read_json("news_sentiment.json")

# Display news articles
st.write(f"## 📢 Latest News for {company_name}")
st.dataframe(news_df[["Title", "Sentiment"]])

# Sentiment Distribution
sentiment_counts = news_df["Sentiment"].value_counts()
st.write("### 📊 Sentiment Distribution")
st.bar_chart(sentiment_counts)

# Generate Hindi TTS Summary
def generate_hindi_tts():
    sentiment_summary = (
        f"इस कंपनी के समाचारों का विश्लेषण किया गया है। "
        f"कुल {len(news_df)} समाचार लेखों में से, "
        f"{sentiment_counts.get('Positive', 0)} सकारात्मक हैं, "
        f"{sentiment_counts.get('Neutral', 0)} तटस्थ हैं, "
        f"और {sentiment_counts.get('Negative', 0)} नकारात्मक हैं।"
    )

    tts = gTTS(text=sentiment_summary, lang="hi")
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    return audio_bytes.getvalue()

# Generate and display audio
st.write("### 🎙️ Hindi Audio Summary")
if st.button("🔊 Generate & Play Audio"):
    audio_data = generate_hindi_tts()
    st.audio(audio_data, format="audio/mp3")
    st.success("✅ Audio generated successfully!")
