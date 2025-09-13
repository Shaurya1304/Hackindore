import os
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from transformers import pipeline

# Load Hugging Face pipelines
category_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
sentiment_analyzer = pipeline("sentiment-analysis")

CATEGORIES = ["Disaster", "Proud", "Happy", "Immoral", "Unethical", "Political", "Neutral"]

API_KEY = os.getenv("API_KEY", "changeme")

app = FastAPI()

class NewsItem(BaseModel):
    title: str
    url: str
    content: str

@app.post("/analyze")
async def analyze_news(item: NewsItem, request: Request):
    # API Key check
    api_key = request.headers.get("x-api-key")
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

    # Category classification
    category_result = category_classifier(item.content, CATEGORIES)
    best_category = category_result["labels"][0]
    category_score = category_result["scores"][0]

    # Sentiment analysis
    sentiment_result = sentiment_analyzer(item.content)[0]
    sentiment_label = sentiment_result["label"]
    sentiment_score = sentiment_result["score"]

    return {
        "first_title": item.title,
        "category": best_category,
        "category_score": round(category_score, 4),
        "sentiment": sentiment_label,
        "sentiment_score": round(sentiment_score, 4)
    }
