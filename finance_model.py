"""
01_collect_and_clean_prices.py

Combined pipeline for price collection.
Executes all steps in a single run:
  1. Downloads raw close prices and volume from Yahoo Finance.
  2. Checks for missing data (>20%).
  3. Patches flagged tickers using Alpha Vantage.
  4. Converts non-EUR tickers to EUR using daily FX rates.
  5. Cleans and finalizes the data (forward-fill and dropna).
"""

import os
import time
import requests
import pandas as pd
import yfinance as yf

# Assuming these are all defined in your config.py
from data_config import (
    ALL_TICKERS, START_DATE, END_DATE, DATA_DIR, 
    RAW_PRICES_FILE, RAW_VOLUME_FILE, BAD_TICKERS_FILE, 
    PATCHED_PRICES_FILE, EUR_PRICES_FILE, CLEAN_PRICES_FILE, 
    ALPHA_VANTAGE_API_KEY, NON_EUR_TICKERS
)

# ==========================================
# STEP 1: Download Raw Data
# ==========================================
def download_prices(tickers, start, end):
    print(f"\n--- STEP 1: Downloading {len(tickers)} tickers from {start} to {end} ---")
    raw = yf.download(
        tickers=list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=True,
        group_by="column",
    )
    if raw.empty:
        raise RuntimeError(
            "No data returned at all. Check your internet connection, "
            "or confirm yfinance still works (Yahoo occasionally changes "
            "its endpoint format)."
        )
    close = raw["Close"]
    volume = raw["Volume"] if "Volume" in raw.columns.get_level_values(0) else pd.DataFrame(index=close.index)
    return close, volume


# ==========================================
# STEP 2: Check Missing Data
# ==========================================
def report_missing(close_prices):
    missing = close_prices.isna().sum()
    missing_pct = (missing / len(close_prices) * 100).round(2)
    report = pd.DataFrame({"missing_days": missing, "missing_pct": missing_pct})
    report = report.sort_values("missing_pct", ascending=False)
    return report


# ==========================================
# STEP 3: Alpha Vantage Fallback
# ==========================================
def fetch_alpha_vantage_prices(ticker, api_key=ALPHA_VANTAGE_API_KEY):
    if api_key == "YOUR_KEY_HERE":
        print("  Skipping Alpha Vantage: no API key set in config.py. "
              "Get a free key at https://www.alphavantage.co/support/#api-key")
        return None
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "outputsize": "full",
        "apikey": api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=20)
        data = resp.json()
        series = data.get("Time Series (Daily)")
        if series is None:
            print(f"  Alpha Vantage returned no data for {ticker}: "
                  f"{data.get('Note', data.get('Error Message', 'unknown reason'))[:150]}")
            return None
        df = pd.DataFrame(series).T
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df["5. adjusted close"].astype(float)
    except Exception as e:
        print(f"  Alpha Vantage fetch failed for {ticker}: {str(e)[:100]}")
        return None


# ==========================================
# STEP 4: Convert to EUR
# ==========================================
def convert_to_eur(close_prices, non_eur_map):
    fx_pairs = set(non_eur_map.values())
    fx_rates = {}
    for pair in fx_pairs:
        print(f"Downloading FX rate: {pair}")
        fx_data = yf.download(pair, start=START_DATE, end=END_DATE,
                               auto_adjust=True, progress=False)
        if fx_data.empty:
            raise RuntimeError(f"Could not download FX rate for {pair}.")
        fx_close = fx_data["Close"]
        fx_rates[pair] = fx_close.iloc[:, 0] if hasattr(fx_close, "iloc") and fx_close.ndim > 1 else fx_close

    converted = close_prices.copy()
    for ticker, fx_pair in non_eur_map.items():
        if ticker not in converted.columns:
            continue
        rate = fx_rates[fx_pair].reindex(converted.index).ffill()
        before_sample = converted[ticker].dropna().iloc[0] if converted[ticker].notna().any() else None
        converted[ticker] = converted[ticker] * rate
        after_sample = converted[ticker].dropna().iloc[0] if converted[ticker].notna().any() else None
        print(f"Converted {ticker} using {fx_pair}: "
              f"first value {before_sample} -> {after_sample} (EUR)")
    return converted


# ==========================================
# STEP 5: Clean and Finalize
# ==========================================
def clean_prices(close_prices):
    filled = close_prices.ffill()
    filled = filled.dropna(how="all")
    return filled


# ==========================================
# MAIN PIPELINE EXECUTION
# ==========================================
def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # --- STEP 1: Download Raw ---
    close_prices, volume = download_prices(ALL_TICKERS, START_DATE, END_DATE)
    close_prices.to_csv(RAW_PRICES_FILE)
    volume.to_csv(RAW_VOLUME_FILE)
    print(f"Saved raw prices to {RAW_PRICES_FILE}")
    print(f"Saved raw volume to {RAW_VOLUME_FILE}")


    # --- STEP 2: Check Missing ---
    print("\n--- STEP 2: Analyzing Missing Data ---")
    report = report_missing(close_prices)
    bad = report[report["missing_pct"] > 20]
    
    if len(bad) > 0:
        print(f"{len(bad)} ticker(s) flagged with >20% missing data:")
        print(bad.to_string())
        bad.to_csv(BAD_TICKERS_FILE)
        print(f"Saved flagged-tickers list to {BAD_TICKERS_FILE}")
    else:
        print("None — every ticker has good coverage.")


    # --- STEP 3: Alpha Vantage Fallback ---
    print("\n--- STEP 3: Alpha Vantage Fallback ---")
    bad_tickers = list(bad.index)
    if not bad_tickers:
        print("Nothing to patch. Passing raw prices through.")
    else:
        print(f"Attempting Alpha Vantage fallback for {len(bad_tickers)} ticker(s)...")
        for ticker in bad_tickers:
            print(f"Fetching {ticker} from Alpha Vantage...")
            av_series = fetch_alpha_vantage_prices(ticker)
            if av_series is not None:
                close_prices[ticker] = av_series.reindex(close_prices.index)
                print(f"  Patched {ticker} using Alpha Vantage")
            time.sleep(15)  # respect the 5-requests-per-minute free-tier limit

    close_prices.to_csv(PATCHED_PRICES_FILE)
    print(f"Saved patched prices to {PATCHED_PRICES_FILE}")


    # --- STEP 4: Convert to EUR ---
    print("\n--- STEP 4: Converting Non-EUR Tickers to EUR ---")
    if NON_EUR_TICKERS:
        close_prices = convert_to_eur(close_prices, NON_EUR_TICKERS)
        close_prices.to_csv(EUR_PRICES_FILE)
        print(f"Saved EUR-converted prices to {EUR_PRICES_FILE}")
    else:
        print("No non-EUR tickers found in this universe (skipping).")


    # --- STEP 5: Clean and Finalize ---
    print("\n--- STEP 5: Cleaning and Finalizing ---")
    missing_before = close_prices.isna().sum().sum()
    print(f"Total missing values before forward-fill: {missing_before}")

    clean = clean_prices(close_prices)
    
    missing_after = clean.isna().sum().sum()
    print(f"Total missing values after forward-fill: {missing_after}")

    clean.to_csv(CLEAN_PRICES_FILE)
    print(f"Saved to: {CLEAN_PRICES_FILE}")

    # ==========================================
    # FINAL REPORT
    # ==========================================
    print("\n" + "="*50)
    print("           PIPELINE COMPLETE")
    print("="*50)
    print(f"Final shape: {clean.shape[0]} days x {clean.shape[1]} assets")
    print(f"Date range:  {clean.index.min()} to {clean.index.max()}")
    print("\nFirst 3 rows:")
    print(clean.head(3))
    print("\nLast 3 rows:")
    print(clean.tail(3))

    if missing_after == 0:
        print("\nPipeline finished successfully. clean_prices_eur.csv is ready.")
    else:
        print(f"\nWARNING: {missing_after} missing values remain even after "
              f"forward-fill. Check these columns:")
        print(clean.isna().sum()[clean.isna().sum() > 0].to_string())


if __name__ == "__main__":
    main()