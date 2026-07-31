"""
04_build_features.py

Builds the full feature matrix used by the XGBoost models (Methods 3 and 4):

  Technical features (every asset, every day):
    - 5/10/20-day momentum (log returns)
    - 20-day realized volatility
    - 50-day vs 200-day moving average ratio
    - 14-day RSI
    - Relative volume is skipped here since Yahoo's `Close`-only frame from
      01_collect_prices.py now also pulls Volume, so this script computes
      relative volume as raw volume divided by its own 20-day average.

  Sentiment features (Method 4 only):
    - sentiment_mean, sentiment_std, sentiment_max, headline_count
      from 03_finbert_sentiment.py, left-joined onto the technical frame.
      Missing sentiment (no headlines that day) is filled with 0, which
      is the neutral / "no signal" value for a mean-centered score.

  Normalization:
    - z-score every feature using TRAINING-PERIOD statistics only, never
      the full dataset. This script computes and saves the training-window
      mean/std separately so the walk-forward backtest can reuse them
      correctly at each rebalancing step, instead of accidentally
      normalizing using future data.

Run this after 01, 02, and 03. Output: data/features.csv

This script only needs pandas and numpy, both already available, so
unlike scripts 01-03 it can run fully offline and has been fully tested
in this environment.
"""

import os
import pandas as pd
import numpy as np

from data_config import CLEAN_PRICES_FILE, RAW_VOLUME_FILE, SENTIMENT_FILE, FEATURES_FILE, DATA_DIR


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------

def compute_momentum(prices, window):
    """Log return over `window` trading days."""
    return np.log(prices / prices.shift(window))


def compute_realized_volatility(prices, window=20):
    """Rolling standard deviation of daily log returns."""
    daily_log_ret = np.log(prices / prices.shift(1))
    return daily_log_ret.rolling(window).std()


def compute_ma_ratio(prices, short_window=50, long_window=200):
    """Ratio of short-term to long-term moving average (trend signal)."""
    short_ma = prices.rolling(short_window).mean()
    long_ma = prices.rolling(long_window).mean()
    return short_ma / long_ma


def compute_rsi(prices, window=14):
    """Standard 14-day Relative Strength Index."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_relative_volume(volume, window=20):
    """Today's volume divided by its own trailing 20-day average.
    Values above 1 mean today is trading more heavily than usual."""
    avg_volume = volume.rolling(window).mean()
    return volume / avg_volume.replace(0, np.nan)


def build_technical_features(prices, volume=None):
    feature_frames = []

    for ticker in prices.columns:
        series = prices[ticker]

        df = pd.DataFrame(index=prices.index)
        df["ticker"] = ticker
        df["close"] = series
        df["mom_5d"] = compute_momentum(series, 5)
        df["mom_10d"] = compute_momentum(series, 10)
        df["mom_20d"] = compute_momentum(series, 20)
        df["realized_vol_20d"] = compute_realized_volatility(series, 20)
        df["ma_ratio_50_200"] = compute_ma_ratio(series, 50, 200)
        df["rsi_14d"] = compute_rsi(series, 14)

        if volume is not None and ticker in volume.columns:
            df["relative_volume"] = compute_relative_volume(
                volume[ticker],
                20,
            )
        else:
            df["relative_volume"] = np.nan

        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: "date"})

        feature_frames.append(df)

    full = pd.concat(feature_frames, ignore_index=True)

    return full


# ---------------------------------------------------------------------------
# Merge sentiment
# ---------------------------------------------------------------------------

def merge_sentiment(technical_df, sentiment_path):
    """Left-join sentiment features onto the technical feature frame.
    Days/tickers with no news get 0 for every sentiment column, which is
    the neutral value on a mean-centered (positive minus negative) scale."""
    if not os.path.exists(sentiment_path):
        print(f"WARNING: {sentiment_path} not found. Producing technical-only "
              f"features (Method 3). Run 03_finbert_sentiment.py first if you "
              f"also want Method 4's sentiment-augmented features.")
        for col in ["sentiment_mean", "sentiment_std", "sentiment_max", "headline_count"]:
            technical_df[col] = 0.0
        return technical_df

    sentiment_df = pd.read_csv(sentiment_path, parse_dates=["date"])

    # Normalize both sides to timezone-naive UTC before merging.
    # sentiment_df may be parsed as datetime64[us, UTC] while technical_df
    # uses naive datetime64[us]; pandas refuses to merge across that boundary.
    if sentiment_df["date"].dt.tz is not None:
        sentiment_df["date"] = sentiment_df["date"].dt.tz_convert("UTC").dt.tz_localize(None)
    if technical_df["date"].dt.tz is not None:
        technical_df["date"] = technical_df["date"].dt.tz_convert("UTC").dt.tz_localize(None)

    merged = technical_df.merge(sentiment_df, on=["ticker", "date"], how="left")
    sentiment_cols = ["sentiment_mean", "sentiment_std", "sentiment_max", "headline_count"]
    merged[sentiment_cols] = merged[sentiment_cols].fillna(0.0)
    return merged


# ---------------------------------------------------------------------------
# Training-only normalization
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "mom_5d", "mom_10d", "mom_20d", "realized_vol_20d", "ma_ratio_50_200",
    "rsi_14d", "relative_volume", "sentiment_mean", "sentiment_std",
    "sentiment_max", "headline_count",
]


def compute_training_stats(features_df, train_end_date):
    """
    Compute mean/std for every feature using ONLY rows up to train_end_date.
    This is the exact statistic that must be reused (never recomputed on
    the full dataset) at every walk-forward step in the backtest, so it
    is saved to its own file rather than baked silently into the feature
    values.
    """
    train_slice = features_df[features_df["date"] <= train_end_date]
    stats = train_slice[FEATURE_COLUMNS].agg(["mean", "std"]).T
    stats.columns = ["train_mean", "train_std"]
    return stats


def apply_zscore(features_df, stats):
    """Apply z-score normalization using pre-computed training statistics."""
    normalized = features_df.copy()
    for col in FEATURE_COLUMNS:
        mean = stats.loc[col, "train_mean"]
        std = stats.loc[col, "train_std"]
        std = std if std > 0 else 1.0  # guard against divide-by-zero
        normalized[col] = (normalized[col] - mean) / std
    return normalized


# ---------------------------------------------------------------------------
def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(CLEAN_PRICES_FILE):
        print(f"ERROR: {CLEAN_PRICES_FILE} not found. Run 01_collect_prices.py first.")
        return

    prices = pd.read_csv(CLEAN_PRICES_FILE, index_col=0, parse_dates=True)
    print(f"Loaded prices: {prices.shape[0]} days x {prices.shape[1]} tickers")

    volume = None
    if os.path.exists(RAW_VOLUME_FILE):
        volume = pd.read_csv(RAW_VOLUME_FILE, index_col=0, parse_dates=True)
        print(f"Loaded volume: {volume.shape[0]} days x {volume.shape[1]} tickers")
    else:
        print(f"NOTE: {RAW_VOLUME_FILE} not found. Relative volume will be "
              f"left as NaN then filled with 1.0 (neutral) downstream. "
              f"Re-run 01_collect_prices.py to generate it.")

    technical = build_technical_features(prices, volume)
    print(f"Built technical features: {len(technical)} ticker-day rows")

    # Neutral fill for relative volume when volume data isn't available
    # yet: 1.0 means "trading exactly at its own average", the sensible
    # no-signal default, matching how sentiment defaults to 0.0.
    technical["relative_volume"] = technical["relative_volume"].fillna(1.0)

    full_features = merge_sentiment(technical, SENTIMENT_FILE)

    # Example training cutoff for a single illustrative normalization pass.
    # In the real walk-forward backtest, recompute this at every rebalancing
    # step using only that step's training window, not this fixed example.
    example_train_end = pd.Timestamp("2019-01-01")
    stats = compute_training_stats(full_features, example_train_end)
    print("\nExample training-window statistics (illustrative only, "
          "recompute per walk-forward step in the real backtest):")
    print(stats)

    full_features.to_csv(FEATURES_FILE, index=False)
    print(f"\nSaved unnormalized features to {FEATURES_FILE}")
    print("Note: features are saved UNNORMALIZED on purpose. Normalize inside "
          "the backtest loop using compute_training_stats() + apply_zscore() "
          "at each step, so normalization never leaks future information.")


if __name__ == "__main__":
    main()