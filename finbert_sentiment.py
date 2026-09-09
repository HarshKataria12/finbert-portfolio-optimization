import os
import pandas as pd
import numpy as np
 
from data_config import NEWS_FILE, SENTIMENT_FILE, DATA_DIR, CLEAN_PRICES_FILE

MARKET_CLOSE_HOUR = 17  # 24-hour clock, LOCAL exchange time (CET/CEST)
NEWS_TIMEZONE = "UTC"        # confirmed: GDELT timestamps are UTC
EXCHANGE_TIMEZONE = "Europe/Berlin"  # covers all euro-area exchanges in this
                                     # universe under one CET/CEST calendar
FINBERT_MODEL = "yiyanghkust/finbert-tone"
US_LISTED_ETFS = {
    "EWI",
    "EWP",
    "EWN",
    "EWL",
}

def load_finbert():
    """Load the pre-trained FinBERT model. Downloads weights on first run.
    Reads the label ordering from the model's own config rather than
    assuming it, since different FinBERT checkpoints order their labels
    differently."""
    from transformers import BertTokenizer, BertForSequenceClassification
    import torch

    print(f"Loading {FINBERT_MODEL} ...")
    tokenizer = BertTokenizer.from_pretrained(FINBERT_MODEL)
    model = BertForSequenceClassification.from_pretrained(FINBERT_MODEL)
    model.eval()

    pos_idx = model.config.label2id["Positive"]
    neg_idx = model.config.label2id["Negative"]
    print(f"  Label mapping confirmed from model config: "
          f"positive=index {pos_idx}, negative=index {neg_idx}")

    return tokenizer, model, torch, pos_idx, neg_idx

def score_headline(headline, tokenizer, model, torch, pos_idx, neg_idx):
    """
    Returns a single sentiment score for one headline:
        P(positive) - P(negative)
    pos_idx/neg_idx come from model.config.label2id, read once in
    load_finbert(), rather than a hardcoded guess at label order.
    """
    inputs = tokenizer(headline, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).squeeze()
    score = probs[pos_idx].item() - probs[neg_idx].item()
    return score

def score_all_headlines(news_df, tokenizer, model, torch, pos_idx, neg_idx, batch_report_every=200):
    """Score every headline in the dataframe, one at a time."""
    scores = []
    for i, headline in enumerate(news_df["headline"]):
        try:
            scores.append(score_headline(str(headline), tokenizer, model, torch, pos_idx, neg_idx))
        except Exception as e:
            print(f"  Failed to score headline at row {i}: {str(e)[:80]}")
            scores.append(np.nan)
        if (i + 1) % batch_report_every == 0:
            print(f"  Scored {i + 1}/{len(news_df)} headlines")
    return scores

def assign_effective_trading_day(publish_time, ticker):
    """Assign news to the trading day when it becomes usable."""

    timestamp = pd.Timestamp(publish_time)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    if ticker in US_LISTED_ETFS:
        local_time = timestamp.tz_convert("America/New_York")
        market_close_minutes = 16 * 60
    else:
        local_time = timestamp.tz_convert("Europe/Berlin")
        market_close_minutes = 17 * 60 + 30

    effective_day = local_time.tz_localize(None).normalize()

    publication_minutes = (
        local_time.hour * 60
        + local_time.minute
    )

    if publication_minutes >= market_close_minutes:
        effective_day += pd.Timedelta(days=1)

    return effective_day


def roll_to_next_trading_day(day, trading_dates):
    """After assign_effective_trading_day, push any date that isn't an
    actual trading day (weekend or holiday) forward to the next one that
    is, so a Friday-evening or weekend headline lands on the correct
    Monday rather than a day with no price data to attach it to."""
    trading_dates = pd.DatetimeIndex(trading_dates)
    pos = trading_dates.searchsorted(day)
    if pos >= len(trading_dates):
        return trading_dates[-1]
    return trading_dates[pos]

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
 
    tokenizer, model, torch, pos_idx, neg_idx = load_finbert()
    
 
    print("Scoring headlines with FinBERT ...")
    news_df["sentiment_score"] = score_all_headlines(news_df, tokenizer, model, torch, pos_idx, neg_idx)
    news_df = news_df.dropna(subset=["sentiment_score"])
 
    print("Applying no-look-ahead rule (assigning effective trading day) ...")
    news_df["effective_day"] = news_df.apply(
    lambda row: assign_effective_trading_day(
        row["publish_time"],
        row["ticker"],
    ),
    axis=1,
)

    prices_for_calendar = pd.read_csv(CLEAN_PRICES_FILE, index_col=0, parse_dates=True)
    trading_days = prices_for_calendar.index
    news_df["effective_day"] = news_df["effective_day"].apply(
        lambda d: roll_to_next_trading_day(d, trading_days))
 
    daily_sentiment = aggregate_daily_sentiment(news_df)
    daily_sentiment.to_csv(SENTIMENT_FILE, index=False)
 
    print(f"\nSaved daily sentiment features to {SENTIMENT_FILE}")
    print(f"Covers {daily_sentiment['ticker'].nunique()} tickers, "
          f"{len(daily_sentiment)} ticker-days total")
 
 
if __name__ == "__main__":
    main()