import streamlit as st
import requests
import pandas as pd
from gtts import gTTS
from io import BytesIO

# Dropdown for company selection
COMPANIES = ["Tesla", "Apple", "Microsoft", "Amazon", "Google"]

st.title("📰 News Sentiment Analyzer")
st.subheader("🔍 Select a company to analyze its latest news sentiment")

company_name = st.selectbox("Choose a company", COMPANIES)

# Fetch news from FastAPI
api_url = f"http://localhost:7860/news?company={company_name}"
response = requests.get(api_url)

if response.status_code == 200:
    news_data = response.json()
    news_df = pd.DataFrame(news_data)

    if not news_df.empty:
        st.write(f"## 📢 Latest News for {company_name}")
        st.dataframe(news_df[["Title", "Summary", "Sentiment", "URL"]])

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

        # Generate and display TTS audio
        st.write("### 🎙️ Hindi Audio Summary")
        if st.button("🔊 Generate & Play Audio"):
            audio_data = generate_hindi_tts()
            st.audio(audio_data, format="audio/mp3")
            st.success("✅ Audio generated successfully!")
    else:
        st.warning("No news found for this company.")
else:
    st.error("Error fetching news. Please try again later.")
