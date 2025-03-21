import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from gtts import gTTS
from io import BytesIO

# List of companies for dropdown
COMPANIES = ["Tesla", "Apple", "Microsoft", "Amazon", "Google"]

st.title("📰 News Sentiment Analyzer")
st.subheader("🔍 Select a company to analyze its latest news sentiment")

# Dropdown for company selection
company_name = st.selectbox("Choose a company", COMPANIES)

# Function to scrape latest news from Bing News
def get_news_articles(company_name):
    """Scrapes latest news for the selected company from Bing News"""
    search_url = f"https://www.bing.com/news/search?q={company_name}&FORM=HDRSC6"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    articles = soup.find_all("a", {"class": "title"})
    summaries = soup.find_all("div", {"class": "snippet"})

    news_data = []
    for i in range(min(len(articles), 10)):  # Limit to 10 articles
        title = articles[i].text.strip()
        summary = summaries[i].text.strip() if i < len(summaries) else "No summary available"
        link = articles[i]["href"]
        news_data.append({"Title": title, "Summary": summary, "URL": link})

    return pd.DataFrame(news_data)

# Get fresh news based on the selected company
news_df = get_news_articles(company_name)

# Display news articles
st.write(f"## 📢 Latest News for {company_name}")
st.dataframe(news_df[["Title", "Summary", "URL"]])

# Sentiment Analysis using VADER
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """Classifies sentiment as Positive, Negative, or Neutral"""
    score = sia.polarity_scores(text)["compound"]
    return "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"

# Apply sentiment analysis
news_df["Sentiment"] = news_df["Summary"].apply(analyze_sentiment)

# Display updated dataframe
st.write(f"### 📊 Sentiment Analysis for {company_name}")
st.dataframe(news_df[["Title", "Sentiment"]])

# Generate Hindi TTS Summary
def generate_hindi_tts():
    sentiment_counts = news_df["Sentiment"].value_counts()
    summary_text = (
        f"इस कंपनी के समाचारों का विश्लेषण किया गया है। "
        f"कुल {len(news_df)} समाचार लेखों में से, "
        f"{sentiment_counts.get('Positive', 0)} सकारात्मक हैं, "
        f"{sentiment_counts.get('Neutral', 0)} तटस्थ हैं, "
        f"और {sentiment_counts.get('Negative', 0)} नकारात्मक हैं।"
    )

    tts = gTTS(text=summary_text, lang="hi")
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    return audio_bytes.getvalue()

# Generate and display TTS audio
st.write("### 🎙️ Hindi Audio Summary")
if st.button("🔊 Generate & Play Audio"):
    audio_data = generate_hindi_tts()
    st.audio(audio_data, format="audio/mp3")
    st.success("✅ Audio generated successfully!")
