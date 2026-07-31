import os
import pandas as pd
import numpy as np
 
from data_config import NEWS_FILE, SENTIMENT_FILE, DATA_DIR

MARKET_CLOSE_HOUR = 17  # 24-hour clock, local exchange time approximation
FINBERT_MODEL = "yiyanghkust/finbert-tone"

def load_finbert():
    """Load the pre-trained FinBERT model. Downloads weights on first run."""
    from transformers import BertTokenizer, BertForSequenceClassification
    import torch
 
    print(f"Loading {FINBERT_MODEL} ...")
    tokenizer = BertTokenizer.from_pretrained(FINBERT_MODEL)
    model = BertForSequenceClassification.from_pretrained(FINBERT_MODEL)
    model.eval()
    return tokenizer, model, torch

def score_headline(headline, tokenizer, model, torch):
    """
    Returns a single sentiment score for one headline:
        P(positive) - P(negative)
    Label order for yiyanghkust/finbert-tone is [positive, negative, neutral].
    """
    inputs = tokenizer(headline, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).squeeze()
    score = probs[0].item() - probs[1].item()
    return score

def score_all_headlines(news_df, tokenizer, model, torch, batch_report_every=200):
    """Score every headline in the dataframe, one at a time."""
    scores = []
    for i, headline in enumerate(news_df["headline"]):
        try:
            scores.append(score_headline(str(headline), tokenizer, model, torch))
        except Exception as e:
            print(f"  Failed to score headline at row {i}: {str(e)[:80]}")
            scores.append(np.nan)
        if (i + 1) % batch_report_every == 0:
            print(f"  Scored {i + 1}/{len(news_df)} headlines")
    return scores

def assign_effective_trading_day(publish_time, market_close_hour=MARKET_CLOSE_HOUR):
    """
    The core no-look-ahead rule.
    If a headline is published before market close, it counts for THAT
    calendar day. If published at or after market close, it counts for
    the NEXT calendar day, since the market couldn't have reacted to it
    yet on the day it was published.
    """
    if publish_time.hour < market_close_hour:
        return publish_time.normalize()
    else:
        return publish_time.normalize() + pd.Timedelta(days=1)

def aggregate_daily_sentiment(scored_news):
    """
    Group by ticker and effective trading day, then compute mean, std,
    and max sentiment score per group, matching Ruan and Jiang (2025).
    """
    grouped = scored_news.groupby(["ticker", "effective_day"])["sentiment_score"]
    daily = grouped.agg(["mean", "std", "max", "count"]).reset_index()
    daily.columns = ["ticker", "date", "sentiment_mean", "sentiment_std",
                      "sentiment_max", "headline_count"]
    # Days with only one headline have no meaningful std; fill with 0
    # rather than leaving NaN, which would otherwise break downstream
    # feature scaling.
    daily["sentiment_std"] = daily["sentiment_std"].fillna(0.0)
    return daily
 
 
def main():
    os.makedirs(DATA_DIR, exist_ok=True)
 
    if not os.path.exists(NEWS_FILE):
        print(f"ERROR: {NEWS_FILE} not found. Run 02_collect_news.py first.")
        return
 
    news_df = pd.read_csv(NEWS_FILE, parse_dates=["publish_time"])
    print(f"Loaded {len(news_df)} headlines from {NEWS_FILE}")
 
    tokenizer, model, torch = load_finbert()
 
    print("Scoring headlines with FinBERT ...")
    news_df["sentiment_score"] = score_all_headlines(news_df, tokenizer, model, torch)
    news_df = news_df.dropna(subset=["sentiment_score"])
 
    print("Applying no-look-ahead rule (assigning effective trading day) ...")
    news_df["effective_day"] = news_df["publish_time"].apply(assign_effective_trading_day)
 
    daily_sentiment = aggregate_daily_sentiment(news_df)
    daily_sentiment.to_csv(SENTIMENT_FILE, index=False)
 
    print(f"\nSaved daily sentiment features to {SENTIMENT_FILE}")
    print(f"Covers {daily_sentiment['ticker'].nunique()} tickers, "
          f"{len(daily_sentiment)} ticker-days total")
 
 
if __name__ == "__main__":
    main()