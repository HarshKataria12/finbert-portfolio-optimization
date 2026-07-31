"""
02_collect_news.py

Collects news headlines for every ticker in the universe:
  - Yahoo Finance's ticker news endpoint (recent coverage, ~1-2 years)
  - GDELT's DOC 2.0 API (historical depth, free, no signup, back to 2015)

Headlines are tagged with the ticker they belong to and the exact
publish timestamp, since the FinBERT script downstream needs that
timestamp to enforce the no-look-ahead rule.

Design decisions:
  - GDELT is queried in quarterly chunks to stay within the 250-record
    per-request cap and maximise historical coverage back to START_DATE.
  - GDELT is skipped entirely for ETFs — ETF product names produce no
    meaningful news signal in a plain-text article database. Sentiment
    for ETFs will correctly default to 0.0 (neutral) downstream.
  - Yahoo Finance does not support historical date parameters. It always
    returns its most recent ~50 articles (~1-2 years back). It is used
    only to supplement GDELT for the most recent period, and results are
    filtered to the configured date range before saving.

Run this after 01_collect_prices.py. Output: data/news_headlines.csv

IMPORTANT: needs real internet access (finance.yahoo.com and
api.gdeltproject.org). Will not run inside a locked-down sandbox.
"""

import os
import time
import requests
import pandas as pd
import yfinance as yf

from data_config import (
    ALL_TICKERS, EQUITIES, ETFS,
    DATA_DIR, NEWS_FILE,
    START_DATE, END_DATE,
)


# ---------------------------------------------------------------------------
# Yahoo Finance
# ---------------------------------------------------------------------------

def get_yahoo_news(ticker, start_date=START_DATE, end_date=END_DATE):
    """
    Pull recent headlines via yfinance and filter to the configured date
    range.

    NOTE: Yahoo Finance does not support historical date parameters — it
    always returns its most recent ~50 articles only (roughly 1-2 years
    back). For full historical coverage back to START_DATE, GDELT is the
    primary source. Yahoo here just supplements with the most recent period.
    """
    start_ts = pd.Timestamp(start_date)
    end_ts   = pd.Timestamp(end_date)
    rows     = []

    try:
        t     = yf.Ticker(ticker)
        items = t.news or []

        if not items:
            print(f"  Yahoo Finance: no headlines returned for {ticker} "
                  f"(may be rate-limited or ticker not recognised)")
            return rows

        for item in items:
            pub_time = pd.to_datetime(
                item.get("providerPublishTime", None), unit="s", errors="coerce"
            )
            # Discard anything outside the pipeline date range
            if pd.isna(pub_time) or not (start_ts <= pub_time <= end_ts):
                continue
            rows.append({
                "ticker":       ticker,
                "headline":     item.get("title", ""),
                "publish_time": pub_time,
                "source":       "yahoo_finance",
            })

    except Exception as e:
        print(f"  Yahoo Finance failed for {ticker}: {str(e)[:100]}")

    return rows


# ---------------------------------------------------------------------------
# GDELT
# ---------------------------------------------------------------------------

def get_gdelt_news(company_name, ticker, start_date, end_date, max_records=250):
    """
    Query GDELT DOC 2.0 API in quarterly chunks to work around the
    250-record per-request cap.

    start_date / end_date format: YYYYMMDDHHMMSS
    Empty quarters (no articles found) are silently skipped — an empty
    body from GDELT is not an error, just no coverage for that period.
    """
    # Build quarterly chunks between start_date and end_date
    start_ts = pd.Timestamp(start_date[:8])
    end_ts   = pd.Timestamp(end_date[:8])
    chunks   = pd.date_range(start=start_ts, end=end_ts, freq="QS")
    periods  = list(zip(chunks, chunks[1:]))
    if len(chunks) > 0:
        periods.append((chunks[-1], end_ts))    # final partial quarter

    url  = "https://api.gdeltproject.org/api/v2/doc/doc"
    rows = []

    for chunk_start, chunk_end in periods:
        params = {
            "query":         company_name,
            "mode":          "artlist",
            "startdatetime": chunk_start.strftime("%Y%m%d%H%M%S"),
            "enddatetime":   chunk_end.strftime("%Y%m%d%H%M%S"),
            "format":        "json",
            "maxrecords":    max_records,
        }
        try:
            resp = requests.get(url, params=params, timeout=20)

            # Non-200 status — log and move on
            if resp.status_code != 200:
                print(f"  GDELT HTTP {resp.status_code} for {company_name} "
                      f"({chunk_start.date()} to {chunk_end.date()}) — skipping")
                time.sleep(2)
                continue

            # Empty body = zero articles for this quarter, not an error
            if not resp.text.strip():
                time.sleep(1)
                continue

            data = resp.json()

            for article in data.get("articles", []):
                # Parse timestamp once; safely strip timezone if present
                ts = pd.to_datetime(article.get("seendate", None), errors="coerce")
                if pd.notna(ts) and ts.tzinfo is not None:
                    ts = ts.tz_convert("UTC").tz_localize(None)

                rows.append({
                    "ticker":       ticker,
                    "headline":     article.get("title", ""),
                    "publish_time": ts,
                    "source":       "gdelt",
                })

            time.sleep(1)   # be polite between chunk requests

        except Exception as e:
            print(f"  GDELT failed for {company_name} "
                  f"({chunk_start.date()} to {chunk_end.date()}): {str(e)[:100]}")
            time.sleep(2)   # back off slightly after any error

    return rows


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def collect_all_news(tickers_dict, use_gdelt=True):
    """
    Loop over every ticker, pull Yahoo recent news, and optionally GDELT
    historical news aligned to the pipeline's START_DATE / END_DATE.

    Company names (not ticker symbols) are used for GDELT queries because
    GDELT matches on plain article text, not stock symbols.

    ETFs are skipped for GDELT — their product names produce no reliable
    news signal in a plain-text article database. Sentiment for ETFs will
    default to 0.0 (neutral) in the feature matrix, which is correct.
    """
    all_rows    = []
    gdelt_start = pd.Timestamp(START_DATE).strftime("%Y%m%d%H%M%S")
    gdelt_end   = pd.Timestamp(END_DATE).strftime("%Y%m%d%H%M%S")

    for ticker, name in tickers_dict.items():
        print(f"\nCollecting news for {ticker} ({name}) ...")

        # Yahoo Finance — all tickers, recent period only
        yahoo_rows = get_yahoo_news(ticker)
        all_rows.extend(yahoo_rows)
        print(f"  Yahoo Finance: {len(yahoo_rows)} headlines")

        # GDELT — equities only, full date range in quarterly chunks
        if use_gdelt:
            if ticker in ETFS:
                print(f"  GDELT: skipped (ETF — no meaningful article-level signal)")
            else:
                gdelt_rows = get_gdelt_news(name, ticker, gdelt_start, gdelt_end)
                all_rows.extend(gdelt_rows)
                print(f"  GDELT: {len(gdelt_rows)} headlines")

        time.sleep(1)   # brief pause between tickers

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def deduplicate_headlines(news_df):
    """
    Drop exact duplicate headlines for the same ticker. Duplicates are
    common when the same story is syndicated across outlets and picked up
    by both Yahoo Finance and GDELT.
    """
    before    = len(news_df)
    news_df   = news_df.drop_duplicates(subset=["ticker", "headline"])
    after     = len(news_df)
    removed   = before - after
    print(f"Deduplication removed {removed:,} duplicate headlines "
          f"({before:,} → {after:,})")
    return news_df


def print_sample_table(news_df, n=20):
    """Print the first n headlines as a readable table."""
    sample                 = news_df.head(n).copy()
    sample["publish_time"] = sample["publish_time"].astype(str).str[:19]
    sample["headline"]     = sample["headline"].str[:60]
    sample.index           = range(1, len(sample) + 1)   # dynamic, never hardcoded
    print(sample[["ticker", "headline", "publish_time", "source"]].to_string())
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    news_df = collect_all_news(ALL_TICKERS, use_gdelt=True)

    if news_df.empty:
        print("\nWARNING: no headlines collected at all. "
              "Check internet access and API availability before proceeding.")
        return

    # Drop rows where publish_time could not be parsed
    before_drop = len(news_df)
    news_df     = news_df.dropna(subset=["publish_time"])
    dropped     = before_drop - len(news_df)
    if dropped:
        print(f"Dropped {dropped:,} rows with unparseable publish_time")

    news_df = deduplicate_headlines(news_df)
    news_df = news_df.sort_values(["ticker", "publish_time"]).reset_index(drop=True)

    news_df.to_csv(NEWS_FILE, index=False)
    print(f"\nSaved {len(news_df):,} headlines to {NEWS_FILE}")

    # Safe date range — guards against NaT from failed parses
    min_date = news_df["publish_time"].min()
    max_date = news_df["publish_time"].max()
    print(f"Date range : "
          f"{min_date.date() if pd.notna(min_date) else 'N/A'}"
          f"  →  "
          f"{max_date.date() if pd.notna(max_date) else 'N/A'}")

    # Sample table
    print("\nSample Headlines:")
    print_sample_table(news_df, n=20)

    # Coverage summary
    coverage = news_df.groupby("ticker").size().sort_values(ascending=False)
    print("Headline count per ticker (top 20):")
    print(coverage.head(20).to_string())

    print("\nTickers with ZERO headlines (check these manually):")
    zero_coverage = [t for t in ALL_TICKERS if t not in coverage.index]
    print(zero_coverage if zero_coverage
          else "None — every ticker got at least one headline")

    print("\nSource breakdown:")
    print(news_df["source"].value_counts().to_string())


if __name__ == "__main__":
    main()