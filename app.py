from fastapi import FastAPI
import requests
from bs4 import BeautifulSoup
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download the VADER lexicon
nltk.download('vader_lexicon')

app = FastAPI()
sia = SentimentIntensityAnalyzer()

# Function to scrape latest news
def get_news_articles(company_name):
    """Fetches latest news articles from Bing News."""
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

# Function to analyze sentiment
def analyze_sentiment(text):
    """Classifies sentiment as Positive, Negative, or Neutral."""
    score = sia.polarity_scores(text)["compound"]
    return "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"

# API Endpoint: Get news articles and sentiment analysis
@app.get("/news")
def fetch_news(company: str):
    news_df = get_news_articles(company)
    
    if news_df.empty:
        return {"message": "No news found for this company."}

    # Apply sentiment analysis
    news_df["Sentiment"] = news_df["Summary"].apply(analyze_sentiment)

    # Convert DataFrame to JSON format
    return news_df.to_dict(orient="records")

# API Health Check
@app.get("/")
def home():
    return {"message": "News Sentiment API is running!"}
