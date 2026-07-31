"""
week34_thesis_screenshots.py
============================
Run this ONE file to generate all clean printed outputs
for your Week 3-4 thesis draft screenshots.

HOW TO USE:
  1. Put this file in your model/ folder (same place as data_config.py)
  2. Run:  python week34_thesis_screenshots.py
  3. Screenshot each section that appears in the terminal

Each section is clearly labelled — screenshot individually or all at once.
"""

import os
import pandas as pd
import numpy as np

# ── path to your data folder ──────────────────────────────────────────────────
from data_config import (
    CLEAN_PRICES_FILE, RAW_VOLUME_FILE, FEATURES_FILE,
    SENTIMENT_FILE, RISKFREE_RATE_FILE, NEWS_FILE,
    ALL_TICKERS, EQUITIES, ETFS, NON_EUR_TICKERS,
    START_DATE, END_DATE
)

SEP  = "=" * 65
SEP2 = "-" * 65


def section(title):
    """Print a clean section header for screenshots."""
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def sub(title):
    print(f"\n{SEP2}")
    print(f"  {title}")
    print(SEP2)


# ══════════════════════════════════════════════════════════════════════════════
# SCREENSHOT 1 — Universe Overview  (data_config.py)
# ══════════════════════════════════════════════════════════════════════════════
section("SCREENSHOT 1 — Asset Universe  (data_config.py)")

print(f"\n  Study period   : {START_DATE}  →  {END_DATE}")
print(f"  Total assets   : {len(ALL_TICKERS)}")
print(f"  ETFs           : {len(ETFS)}")
print(f"  Equities       : {len(EQUITIES)}")
print(f"  Non-EUR assets : {len(NON_EUR_TICKERS)}  (CHF → EUR converted)")

sub("ETF Universe")
for ticker, name in ETFS.items():
    print(f"  {ticker:<14}  {name}")

sub("Equity Universe (first 10)")
for ticker, name in list(EQUITIES.items())[:10]:
    print(f"  {ticker:<14}  {name}")
print(f"  ... and {len(EQUITIES)-10} more equities")

sub("Non-EUR Conversion Pairs")
for ticker, fx in NON_EUR_TICKERS.items():
    print(f"  {ticker:<14}  converted via  {fx}")


# ══════════════════════════════════════════════════════════════════════════════
# SCREENSHOT 2 — Clean Prices Summary  (finance_model.py output)
# ══════════════════════════════════════════════════════════════════════════════
section("SCREENSHOT 2 — Clean Price Data  (finance_model.py)")

if os.path.exists(CLEAN_PRICES_FILE):
    prices = pd.read_csv(CLEAN_PRICES_FILE, index_col=0, parse_dates=True)

    print(f"\n  Shape          : {prices.shape[0]} trading days  ×  {prices.shape[1]} assets")
    print(f"  Date range     : {prices.index.min().date()}  →  {prices.index.max().date()}")
    print(f"  Missing values : {prices.isna().sum().sum()}  (after forward-fill)")

    sub("Price Statistics — first 8 tickers (EUR)")
    print(prices.iloc[:, :8].describe().round(2).to_string())

    sub("Sample: Last 5 rows — first 6 tickers")
    print(prices.iloc[-5:, :6].round(2).to_string())

    sub("Missing Data Report (top 10 most missing)")
    missing = prices.isna().mean().mul(100).round(2).sort_values(ascending=False)
    print(missing.head(10).rename("missing_%").to_string())
else:
    print("\n  ⚠  clean_prices_eur.csv not found — run finance_model.py first")


# ══════════════════════════════════════════════════════════════════════════════
# SCREENSHOT 3 — Volume Data  (finance_model.py output)
# ══════════════════════════════════════════════════════════════════════════════
section("SCREENSHOT 3 — Volume Data  (finance_model.py)")

if os.path.exists(RAW_VOLUME_FILE):
    volume = pd.read_csv(RAW_VOLUME_FILE, index_col=0, parse_dates=True)
    print(f"\n  Shape     : {volume.shape[0]} days  ×  {volume.shape[1]} tickers")
    print(f"\n  Average daily volume (millions) — top 10 most active:")
    avg_vol = volume.mean().div(1e6).round(2).sort_values(ascending=False)
    print(avg_vol.head(10).rename("avg_vol_M").to_string())
else:
    print("\n  ⚠  raw_volume.csv not found — run finance_model.py first")


# ══════════════════════════════════════════════════════════════════════════════
# SCREENSHOT 4 — News Headlines  (news_collection.py output)
# ══════════════════════════════════════════════════════════════════════════════
section("SCREENSHOT 4 — News Headlines  (news_collection.py)")

if os.path.exists(NEWS_FILE):
    news = pd.read_csv(NEWS_FILE, parse_dates=["publish_time"])

    print(f"\n  Total headlines    : {len(news):,}")
    print(f"  Unique tickers     : {news['ticker'].nunique()}")
    print(f"  Sources            : {news['source'].value_counts().to_dict()}")
    print(f"  Date range         : {news['publish_time'].min().date()}  →  "
          f"{news['publish_time'].max().date()}")

    sub("Top 10 Tickers by Headline Count")
    coverage = news.groupby("ticker").size().sort_values(ascending=False)
    print(coverage.head(10).rename("headlines").to_string())

    sub("Sample Headlines (10 rows)")
    sample = news.head(10)[["ticker","headline","publish_time","source"]].copy()
    sample["headline"]     = sample["headline"].str[:55]
    sample["publish_time"] = sample["publish_time"].astype(str).str[:16]
    sample.index = range(1, 11)
    print(sample.to_string())
else:
    print("\n  ⚠  news_headlines.csv not found — run news_collection.py first")


# ══════════════════════════════════════════════════════════════════════════════
# SCREENSHOT 5 — FinBERT Sentiment  (finbert_sentiment.py output)
# ══════════════════════════════════════════════════════════════════════════════
section("SCREENSHOT 5 — FinBERT Sentiment Scores  (finbert_sentiment.py)")

if os.path.exists(SENTIMENT_FILE):
    sent = pd.read_csv(SENTIMENT_FILE, parse_dates=["date"])

    print(f"\n  Total ticker-days  : {len(sent):,}")
    print(f"  Unique tickers     : {sent['ticker'].nunique()}")
    print(f"  Date range         : {sent['date'].min().date()}  →  "
          f"{sent['date'].max().date()}")

    sub("Sentiment Score Distribution")
    desc = sent["sentiment_mean"].describe().round(4)
    print(desc.to_string())

    neg = (sent["sentiment_mean"] < -0.1).sum()
    neu = ((sent["sentiment_mean"] >= -0.1) & (sent["sentiment_mean"] <= 0.1)).sum()
    pos = (sent["sentiment_mean"] > 0.1).sum()
    total = len(sent)
    print(f"\n  Negative  (score < -0.1)  : {neg:>5}  ({neg/total*100:.1f}%)")
    print(f"  Neutral   (-0.1 to 0.1)  : {neu:>5}  ({neu/total*100:.1f}%)")
    print(f"  Positive  (score > 0.1)  : {pos:>5}  ({pos/total*100:.1f}%)")

    sub("Per-Ticker Average Sentiment (top 10 most positive)")
    top = sent.groupby("ticker")["sentiment_mean"].mean().sort_values(ascending=False)
    print(top.head(10).round(4).rename("avg_sentiment").to_string())
else:
    print("\n  ⚠  daily_sentiment.csv not found — run finbert_sentiment.py first")


# ══════════════════════════════════════════════════════════════════════════════
# SCREENSHOT 6 — Feature Matrix  (build_features.py output)
# ══════════════════════════════════════════════════════════════════════════════
section("SCREENSHOT 6 — Feature Matrix  (build_features.py)")

FEATURE_COLS = [
    "mom_5d","mom_10d","mom_20d",
    "realized_vol_20d","ma_ratio_50_200",
    "rsi_14d","relative_volume",
    "sentiment_mean","sentiment_std",
    "sentiment_max","headline_count",
]

if os.path.exists(FEATURES_FILE):
    feat = pd.read_csv(FEATURES_FILE, parse_dates=["date"])

    print(f"\n  Total rows (ticker-days) : {len(feat):,}")
    print(f"  Unique tickers           : {feat['ticker'].nunique()}")
    print(f"  Total features           : {len(FEATURE_COLS)}")
    print(f"  Date range               : {feat['date'].min().date()}  →  "
          f"{feat['date'].max().date()}")

    sub("Feature Summary Statistics")
    available = [c for c in FEATURE_COLS if c in feat.columns]
    print(feat[available].describe().round(4).to_string())

    sub("Sample Rows — SAP.DE")
    sample = feat[feat["ticker"] == "SAP.DE"].tail(5)
    cols_show = ["date","ticker"] + available[:6]
    print(sample[cols_show].round(4).to_string(index=False))

    sub("Missing Values per Feature")
    missing_feat = feat[available].isna().sum().rename("missing_count")
    missing_feat_pct = (feat[available].isna().mean() * 100).round(2).rename("missing_%")
    print(pd.concat([missing_feat, missing_feat_pct], axis=1).to_string())
else:
    print("\n  ⚠  features.csv not found — run build_features.py first")


# ══════════════════════════════════════════════════════════════════════════════
# SCREENSHOT 7 — Risk-Free Rate  (collect_riskfree_rate.py output)
# ══════════════════════════════════════════════════════════════════════════════
section("SCREENSHOT 7 — Euribor Risk-Free Rate  (collect_riskfree_rate.py)")

if os.path.exists(RISKFREE_RATE_FILE):
    rf = pd.read_csv(RISKFREE_RATE_FILE, parse_dates=["date"])

    print(f"\n  Total daily observations : {len(rf):,}")
    print(f"  Date range               : {rf['date'].min().date()}  →  "
          f"{rf['date'].max().date()}")
    print(f"  Latest rate              : {rf['euribor_3m_rate'].iloc[-1]:.3f}%")
    print(f"  Min rate                 : {rf['euribor_3m_rate'].min():.3f}%"
          f"  (on {rf.loc[rf['euribor_3m_rate'].idxmin(),'date'].date()})")
    print(f"  Max rate                 : {rf['euribor_3m_rate'].max():.3f}%"
          f"  (on {rf.loc[rf['euribor_3m_rate'].idxmax(),'date'].date()})")

    sub("Sample — first and last 3 rows")
    show = pd.concat([rf.head(3), rf.tail(3)])
    show["euribor_3m_rate"] = show["euribor_3m_rate"].round(3)
    print(show.to_string(index=False))
else:
    print("\n  ⚠  euribor_3m.csv not found — run collect_riskfree_rate.py first")


# ══════════════════════════════════════════════════════════════════════════════
# SCREENSHOT 8 — Pipeline Summary (all files)
# ══════════════════════════════════════════════════════════════════════════════
section("SCREENSHOT 8 — Full Pipeline File Check (Week 3-4 Complete)")

files = {
    "clean_prices_eur.csv  (finance_model.py)"    : CLEAN_PRICES_FILE,
    "raw_volume.csv        (finance_model.py)"    : RAW_VOLUME_FILE,
    "news_headlines.csv    (news_collection.py)"  : NEWS_FILE,
    "daily_sentiment.csv   (finbert_sentiment.py)": SENTIMENT_FILE,
    "features.csv          (build_features.py)"   : FEATURES_FILE,
    "euribor_3m.csv        (collect_riskfree_rate)": RISKFREE_RATE_FILE,
}

all_ok = True
print()
for label, path in files.items():
    if os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024
        print(f"  ✅  {label:<50}  {size_kb:>8.1f} KB")
    else:
        print(f"  ❌  {label:<50}  NOT FOUND")
        all_ok = False

print()
if all_ok:
    print("  ✅  ALL FILES PRESENT — Week 3-4 data pipeline is complete")
else:
    print("  ⚠   Some files missing — run the relevant scripts first")

print(f"\n{SEP}\n")