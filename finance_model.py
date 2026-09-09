"""
finance_model.py

General price collection and cleaning pipeline.

Steps:
  1. Load and restrict existing data to the configured date window.
  2. Incrementally download adjusted prices and volume from Yahoo,
     in recursive batches of BATCH_SIZE tickers at a time.
  3. Retry missing assets individually.
  4. Validate recent coverage.
  5. Convert non-EUR assets into EUR.
  6. Forward-fill normal market-calendar gaps.
  7. Save the completed model dataset.

The complete asset universe comes from ALL_TICKERS in data_config.py.
"""

import os
import time
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from data_config import (
    ALL_TICKERS,
    ETFS,
    EQUITIES,
    START_DATE,
    END_DATE,
    DATA_DIR,
    RAW_PRICES_FILE,
    RAW_VOLUME_FILE,
    BAD_TICKERS_FILE,
    PATCHED_PRICES_FILE,
    EUR_PRICES_FILE,
    CLEAN_PRICES_FILE,
    NON_EUR_TICKERS,
)


# ============================================================
# Configuration
# ============================================================

DOWNLOAD_START_DATE = os.environ.get(
    "PRICE_DOWNLOAD_START_DATE",
    START_DATE,
)

DOWNLOAD_END_DATE = os.environ.get(
    "PRICE_DOWNLOAD_END_DATE",
    END_DATE,
)

INCREMENTAL_OVERLAP_DAYS = 14
MISSING_THRESHOLD_PCT = 20.0
MAX_STALENESS_DAYS = 10

# Only one retry. Repeated requests make Yahoo 429 errors worse.
YAHOO_MAX_RETRIES = 1
YAHOO_REQUEST_PAUSE_SECONDS = 5

# If a ticker gets rate-limited during the per-ticker retry loop, it is
# skipped (left for a later run) rather than aborting the whole loop —
# a single problem ticker (bad date range, delisted, etc.) shouldn't
# block every other ticker behind it from being attempted. Only a run
# of several consecutive 429s in a row — a real sign the connection
# itself is locked out, not just one symbol — stops the loop early.
MAX_CONSECUTIVE_RATE_LIMIT_HITS = 3
RATE_LIMIT_BACKOFF_SECONDS = 30

# Batch size for the recursive full-universe download. Keeping each
# yf.download() call to a modest chunk avoids the single giant request
# that is most likely to time out or come back completely empty.
BATCH_SIZE = 50

# Alpha Vantage is used only as a final fallback for these explicitly
# approved tickers. It is never called for the rest of ALL_TICKERS.
ALPHA_FALLBACK_TICKERS = {"REP.MC"}
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"

# Full 2016-2025 history should contain roughly 2,500 sessions. This
# coverage check prevents Alpha Vantage's compact 100-row response from
# being mistaken for a complete history and merged into the thesis data.
ALPHA_MIN_HISTORY_COVERAGE = 0.80


# ============================================================
# Date and dataframe helpers
# ============================================================

def validate_date_configuration():
    """Validate the configured research dates."""
    start_date = pd.Timestamp(DOWNLOAD_START_DATE)
    end_date = pd.Timestamp(DOWNLOAD_END_DATE)

    if start_date >= end_date:
        raise ValueError(
            f"Invalid date configuration: "
            f"start={start_date.date()} must be before "
            f"end={end_date.date()}."
        )

    print(
        f"Configured data window: "
        f"{start_date.date()} to "
        f"{end_date.date()} (end exclusive)"
    )


def normalise_index(frame):
    """Create a sorted, timezone-naive DatetimeIndex."""
    result = frame.copy()

    if result.empty:
        return result

    result.index = pd.DatetimeIndex(
        pd.to_datetime(result.index)
    ).tz_localize(None)

    result = result.loc[~result.index.duplicated(keep="last")]

    return result.sort_index()


def restrict_to_requested_window(frame):
    """Remove observations outside the configured research period."""
    if frame.empty:
        return frame

    result = normalise_index(frame)

    start_date = pd.Timestamp(DOWNLOAD_START_DATE)
    end_date = pd.Timestamp(DOWNLOAD_END_DATE)

    return result.loc[(result.index >= start_date) & (result.index < end_date)]


def read_existing_file(path):
    """Read and restrict an existing time-series CSV."""
    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame()

    return restrict_to_requested_window(frame)


def to_unix_timestamp(value):
    """Convert a date into a UTC Unix timestamp."""
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return int(timestamp.timestamp())


# ============================================================
# Yahoo batch download
# ============================================================

def extract_yahoo_field(raw, field, tickers):
    """Extract Close or Volume from yfinance output."""
    tickers = list(tickers)

    if raw.empty:
        return pd.DataFrame(columns=tickers, dtype=float)

    if isinstance(raw.columns, pd.MultiIndex):
        first_level = raw.columns.get_level_values(0)

        if field not in first_level:
            return pd.DataFrame(index=raw.index, columns=tickers, dtype=float)

        result = raw[field].copy()

    else:
        if field not in raw.columns:
            return pd.DataFrame(index=raw.index, columns=tickers, dtype=float)

        result = raw[field].copy()

    if isinstance(result, pd.Series):
        result = result.to_frame(name=tickers[0])

    return result.reindex(columns=tickers)


def download_prices(tickers, start, end, progress=True):
    """
    Download adjusted prices and volume from Yahoo.

    threads=False avoids sending a large burst of parallel requests.
    """
    tickers = list(tickers)

    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end)

    if start_date >= end_date:
        raise ValueError(
            f"Yahoo download start "
            f"{start_date.date()} must be before "
            f"end {end_date.date()}."
        )

    print(
        f"Yahoo download: "
        f"{len(tickers)} ticker(s), "
        f"{start_date.date()} to "
        f"{end_date.date()}"
    )

    raw = yf.download(
        tickers=tickers,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=progress,
        group_by="column",
        threads=False,
        ignore_tz=True,
        timeout=30,
    )

    if raw.empty:
        raise RuntimeError("Yahoo returned an empty batch.")

    close = extract_yahoo_field(raw, "Close", tickers)
    volume = extract_yahoo_field(raw, "Volume", tickers)

    return (
        restrict_to_requested_window(close),
        restrict_to_requested_window(volume),
    )


# ============================================================
# Recursive batch download
#
# Build the full ticker list once, download it BATCH_SIZE tickers at a
# time, drop whichever tickers came back with usable data from the
# pending list, and recurse on what's left until the pending list is
# empty. Tickers that fail are requeued once (after the rest of the
# universe has had a turn) rather than retried immediately in a tight
# loop, since hammering Yahoo repeatedly for the same failing ticker is
# what triggers 429 rate-limiting. If a full pass makes no further
# progress at all, recursion stops there and whatever is left over is
# handed to the individual per-ticker retry path in main() (which also
# has the direct-Yahoo chart-endpoint fallback).
# ============================================================

def download_batch_recursive(
    pending_tickers,
    close_prices,
    volume,
    batch_size=BATCH_SIZE,
    configured_tickers=None,
):
    """
    Recursively download tickers in chunks until the pending list is empty.

    configured_tickers is the full universe used to shape the saved
    checkpoint files (defaults to ALL_TICKERS from data_config.py). It
    is kept as an explicit parameter — rather than reading the module
    global directly inside the recursive body — so this function stays
    self-contained and testable against any ticker list, not only the
    real thesis universe.
    """
    pending_tickers = list(pending_tickers)

    if configured_tickers is None:
        configured_tickers = list(ALL_TICKERS)

    if not pending_tickers:
        print("\nPending list is empty. Batch download complete.")
        return close_prices, volume

    batch = pending_tickers[:batch_size]
    remaining = pending_tickers[batch_size:]

    print(
        f"\nDownloading batch of {len(batch)} ticker(s); "
        f"{len(remaining)} still queued behind this batch."
    )

    try:
        fresh_close, fresh_volume = download_prices(
            tickers=batch,
            start=DOWNLOAD_START_DATE,
            end=DOWNLOAD_END_DATE,
        )
    except Exception as error:
        print(f"Batch download failed outright: {str(error)[:200]}")
        fresh_close = pd.DataFrame(columns=batch)
        fresh_volume = pd.DataFrame(columns=batch)

    close_prices = fresh_close.combine_first(close_prices)
    volume = fresh_volume.combine_first(volume)

    close_prices, volume = save_download_checkpoint(
        close_prices,
        volume,
        configured_tickers,
    )

    succeeded = [
        ticker for ticker in batch
        if ticker in close_prices.columns and close_prices[ticker].notna().any()
    ]
    failed = [ticker for ticker in batch if ticker not in succeeded]

    print(f"Batch result: {len(succeeded)} succeeded, {len(failed)} failed.")

    if failed:
        print(f"Requeuing for a later pass: {', '.join(failed)}")

    # Failed tickers go to the back of the queue instead of being
    # retried right away.
    next_pending = remaining + failed

    # Stop if this pass recovered nothing and the queue is unchanged
    # apart from order — otherwise a ticker that will never succeed via
    # this path would recurse forever, re-hitting Yahoo each time.
    if not succeeded and set(next_pending) == set(pending_tickers):
        print(
            "No progress on this pass; stopping batch recursion.\n"
            f"Unresolved via batch download: {', '.join(failed)}\n"
            "These will be picked up by the individual retry step."
        )
        return close_prices, volume

    if next_pending:
        time.sleep(YAHOO_REQUEST_PAUSE_SECONDS)

    return download_batch_recursive(
        next_pending,
        close_prices,
        volume,
        batch_size,
        configured_tickers,
    )


# ============================================================
# Incremental update
# ============================================================

def get_incremental_start(existing_path):
    """Choose a valid start date within the requested window."""
    start_boundary = pd.Timestamp(DOWNLOAD_START_DATE)
    end_boundary = pd.Timestamp(DOWNLOAD_END_DATE)

    existing = read_existing_file(existing_path)

    if existing.empty:
        return start_boundary.strftime("%Y-%m-%d")

    last_saved_date = pd.Timestamp(existing.index.max())

    incremental_start = max(
        start_boundary,
        last_saved_date - pd.Timedelta(days=INCREMENTAL_OVERLAP_DAYS),
    )

    if incremental_start >= end_boundary:
        incremental_start = max(
            start_boundary,
            end_boundary - pd.Timedelta(days=INCREMENTAL_OVERLAP_DAYS),
        )

    if incremental_start >= end_boundary:
        raise ValueError(
            f"Could not create a valid incremental "
            f"window: start={incremental_start.date()}, "
            f"end={end_boundary.date()}."
        )

    return incremental_start.strftime("%Y-%m-%d")


def merge_with_existing(existing_path, fresh_data):
    """Merge new data with stored history; new values take priority."""
    existing = read_existing_file(existing_path)
    fresh_data = restrict_to_requested_window(fresh_data)

    if existing.empty:
        return fresh_data

    combined = fresh_data.combine_first(existing)

    return restrict_to_requested_window(combined)


# ============================================================
# Direct Yahoo fallback
# ============================================================

def download_yahoo_chart(ticker, start, end):
    """
    Download adjusted prices through Yahoo's direct chart endpoint.

    Only one host is attempted when Yahoo returns HTTP 429.
    """
    period1 = to_unix_timestamp(start)
    period2 = to_unix_timestamp(end)

    encoded_ticker = quote(ticker, safe="")

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded_ticker}"

    params = {
        "period1": period1,
        "period2": period2,
        "interval": "1d",
        "events": "div,splits",
        "includeAdjustedClose": "true",
    }

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }

    response = requests.get(url, params=params, headers=headers, timeout=30)

    if response.status_code == 429:
        raise RuntimeError(
            "Yahoo rate limit reached (HTTP 429). "
            "Stop the script and wait before rerunning."
        )

    response.raise_for_status()

    payload = response.json()
    chart = payload.get("chart", {})

    chart_error = chart.get("error")
    results = chart.get("result")

    if chart_error:
        raise RuntimeError(chart_error.get("description", str(chart_error)))

    if not results:
        raise RuntimeError(f"Yahoo chart returned no result for {ticker}.")

    result = results[0]

    timestamps = result.get("timestamp", [])
    indicators = result.get("indicators", {})

    if not timestamps:
        raise RuntimeError(f"Yahoo returned no dates for {ticker}.")

    quote_data = indicators.get("quote", [{}])[0]
    adjusted_data = indicators.get("adjclose", [])

    if adjusted_data and adjusted_data[0].get("adjclose") is not None:
        close_values = adjusted_data[0]["adjclose"]
    else:
        close_values = quote_data.get("close", [])

    volume_values = quote_data.get("volume")

    if volume_values is None:
        volume_values = [np.nan] * len(timestamps)

    if len(close_values) != len(timestamps):
        raise RuntimeError(f"Invalid close response for {ticker}.")

    if len(volume_values) != len(timestamps):
        volume_values = [np.nan] * len(timestamps)

    dates = (
        pd.to_datetime(timestamps, unit="s", utc=True)
        .tz_convert(None)
        .normalize()
    )

    close = pd.DataFrame(
        {ticker: pd.to_numeric(close_values, errors="coerce")},
        index=dates,
    )

    volume = pd.DataFrame(
        {ticker: pd.to_numeric(volume_values, errors="coerce")},
        index=dates,
    )

    close = restrict_to_requested_window(close).dropna(subset=[ticker])
    volume = restrict_to_requested_window(volume)

    if close.empty:
        raise RuntimeError(f"No usable direct Yahoo prices were returned for {ticker}.")

    print(f"Direct Yahoo recovered {len(close):,} observations for {ticker}.")

    return close, volume


# ============================================================
# Alpha Vantage fallback (REP.MC only)
# ============================================================

def alpha_vantage_request(params):
    """Make one Alpha Vantage request and surface API error messages."""
    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "ALPHA_VANTAGE_API_KEY is not set. Export the key before "
            "running this script."
        )

    request_params = dict(params)
    request_params["apikey"] = api_key

    response = requests.get(
        ALPHA_VANTAGE_URL,
        params=request_params,
        timeout=60,
    )
    response.raise_for_status()

    payload = response.json()

    for error_key in ("Error Message", "Information", "Note"):
        if payload.get(error_key):
            raise RuntimeError(
                f"Alpha Vantage: {payload[error_key]}"
            )

    return payload


def resolve_alpha_vantage_symbol(ticker):
    """
    Resolve REP.MC to Alpha Vantage's symbol format.

    Set ALPHA_REP_SYMBOL to bypass search when you already know the exact
    Alpha Vantage symbol. Automatic selection requires a Repsol equity
    result for Spain, so a different exchange is never used silently.
    """
    if ticker != "REP.MC":
        raise RuntimeError(
            f"No Alpha Vantage symbol rule is configured for {ticker}."
        )

    override = os.environ.get("ALPHA_REP_SYMBOL", "").strip()

    if override:
        print(
            f"Using ALPHA_REP_SYMBOL={override} as the Alpha Vantage "
            f"mapping for {ticker}."
        )
        return override

    print("Searching Alpha Vantage for Repsol's Spanish listing...")

    payload = alpha_vantage_request({
        "function": "SYMBOL_SEARCH",
        "keywords": "Repsol",
    })

    matches = payload.get("bestMatches", [])

    candidates = []
    for match in matches:
        name = str(match.get("2. name", ""))
        asset_type = str(match.get("3. type", ""))
        region = str(match.get("4. region", ""))

        if (
            "repsol" in name.lower()
            and "equity" in asset_type.lower()
            and "spain" in region.lower()
        ):
            candidates.append(match)

    if not candidates:
        raise RuntimeError(
            "Alpha Vantage SYMBOL_SEARCH did not return a Spanish Repsol "
            "equity. Set ALPHA_REP_SYMBOL to the exact supported symbol "
            "shown by your Alpha Vantage account."
        )

    best_match = max(
        candidates,
        key=lambda item: float(item.get("9. matchScore", 0.0)),
    )

    alpha_symbol = str(best_match.get("1. symbol", "")).strip()

    if not alpha_symbol:
        raise RuntimeError(
            "Alpha Vantage returned a Repsol match without a symbol."
        )

    print(
        f"Alpha Vantage mapping: {ticker} -> {alpha_symbol} "
        f"({best_match.get('4. region', 'unknown region')}, "
        f"{best_match.get('8. currency', 'unknown currency')})."
    )

    return alpha_symbol


def download_one_alpha_vantage_history(ticker, start, end):
    """
    Download complete adjusted daily history for one approved ticker.

    The adjusted endpoint and outputsize=full may require an Alpha Vantage
    premium key. A compact or incomplete response is rejected rather than
    contaminating the saved price panel.
    """
    if ticker not in ALPHA_FALLBACK_TICKERS:
        raise RuntimeError(
            f"Alpha Vantage fallback is disabled for {ticker}."
        )

    alpha_symbol = resolve_alpha_vantage_symbol(ticker)

    print(
        f"Trying Alpha Vantage adjusted full history for {ticker} "
        f"using {alpha_symbol}..."
    )

    payload = alpha_vantage_request({
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": alpha_symbol,
        "outputsize": "full",
    })

    daily = payload.get("Time Series (Daily)")

    if not daily:
        raise RuntimeError(
            "Alpha Vantage returned no adjusted daily time series. "
            "The full adjusted endpoint may require a premium key."
        )

    rows = []
    for date_text, values in daily.items():
        rows.append({
            "date": pd.Timestamp(date_text),
            "close": pd.to_numeric(
                values.get("5. adjusted close"),
                errors="coerce",
            ),
            "volume": pd.to_numeric(
                values.get("6. volume"),
                errors="coerce",
            ),
        })

    history = (
        pd.DataFrame(rows)
        .set_index("date")
        .sort_index()
    )

    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end)
    history = history.loc[
        (history.index >= start_date)
        & (history.index < end_date)
    ]
    history = history.dropna(subset=["close"])

    expected_business_days = len(
        pd.bdate_range(
            start=start_date,
            end=end_date - pd.Timedelta(days=1),
        )
    )
    minimum_rows = max(
        100,
        int(expected_business_days * ALPHA_MIN_HISTORY_COVERAGE),
    )

    if len(history) < minimum_rows:
        raise RuntimeError(
            f"Alpha Vantage returned only {len(history):,} usable rows; "
            f"at least {minimum_rows:,} are required for the configured "
            f"window. Refusing to save incomplete history."
        )

    close = history["close"].rename(ticker).to_frame()
    volume = history["volume"].rename(ticker).to_frame()

    close = restrict_to_requested_window(close)
    volume = restrict_to_requested_window(volume)

    print(
        f"Alpha Vantage recovered {len(close):,} observations for "
        f"{ticker}: {close.index.min().date()} to "
        f"{close.index.max().date()}."
    )

    return close, volume


def download_one_history_with_fallback(ticker, start, end):
    """Try Yahoo first, then Alpha Vantage only for approved tickers."""
    try:
        return download_one_yahoo_history(
            ticker=ticker,
            start=start,
            end=end,
        )
    except Exception as yahoo_error:
        if ticker not in ALPHA_FALLBACK_TICKERS:
            raise

        print(
            f"Yahoo failed for {ticker}. Trying the configured "
            f"Alpha Vantage fallback for this ticker only..."
        )

        try:
            return download_one_alpha_vantage_history(
                ticker=ticker,
                start=start,
                end=end,
            )
        except Exception as alpha_error:
            raise RuntimeError(
                f"Yahoo failed ({str(yahoo_error)[:140]}). "
                f"Alpha Vantage also failed ({str(alpha_error)[:240]})."
            ) from alpha_error


# ============================================================
# Individual Yahoo retry
# ============================================================

def download_one_yahoo_history(ticker, start, end):
    """
    Try yfinance Ticker.history, then fall back to one direct Yahoo
    chart-endpoint request.
    """
    try:
        history = yf.Ticker(ticker).history(
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            actions=False,
            repair=True,
            timeout=30,
        )

        if history.empty or "Close" not in history.columns:
            raise RuntimeError("Ticker.history returned no prices.")

        history.index = pd.DatetimeIndex(
            pd.to_datetime(history.index)
        ).tz_localize(None)

        close = history["Close"].rename(ticker).to_frame()

        if "Volume" in history.columns:
            volume = history["Volume"].rename(ticker).to_frame()
        else:
            volume = pd.DataFrame(index=history.index, columns=[ticker], dtype=float)

        close = restrict_to_requested_window(close)
        volume = restrict_to_requested_window(volume)

        if close[ticker].dropna().empty:
            raise RuntimeError("Ticker.history returned no usable prices.")

        print(
            f"Ticker.history recovered "
            f"{close[ticker].notna().sum():,} "
            f"observations for {ticker}."
        )

        return close, volume

    except Exception as error:
        print(f"Ticker.history failed for {ticker}: {str(error)[:160]}")
        print("Trying one direct Yahoo request...")

    return download_yahoo_chart(ticker=ticker, start=start, end=end)


def retry_yahoo_individually(close_prices, volume, tickers):
    """Retry only assets that failed the main batch download."""
    repaired_close = close_prices.copy()
    repaired_volume = volume.copy()

    if not tickers:
        print("No individual retries are required.")
        return repaired_close, repaired_volume

    print(f"Retrying {len(tickers)} ticker(s) individually...")

    for position, ticker in enumerate(tickers, start=1):
        print(f"{position}/{len(tickers)} {ticker}")

        try:
            one_close, one_volume = download_one_history_with_fallback(
                ticker=ticker,
                start=DOWNLOAD_START_DATE,
                end=DOWNLOAD_END_DATE,
            )

            repaired_close = one_close.combine_first(repaired_close)
            repaired_volume = one_volume.combine_first(repaired_volume)

            print(f"Successfully recovered {ticker}.")

        except Exception as error:
            print(f"WARNING: Could not recover {ticker}: {str(error)[:200]}")

            if "429" in str(error):
                print(
                    "Yahoo has rate-limited this connection. "
                    "Do not rerun immediately."
                )

        if position < len(tickers):
            time.sleep(YAHOO_REQUEST_PAUSE_SECONDS)

    repaired_close = restrict_to_requested_window(repaired_close)
    repaired_volume = restrict_to_requested_window(repaired_volume)

    return repaired_close, repaired_volume


# ============================================================
# Coverage checks
# ============================================================

def report_missing(close_prices):
    """Create missing-data and staleness statistics."""
    if close_prices.empty:
        raise RuntimeError("The price dataframe is empty.")

    latest_market_date = pd.Timestamp(close_prices.index.max())

    rows = []

    for ticker in close_prices.columns:
        series = close_prices[ticker]

        first_valid = series.first_valid_index()
        last_valid = series.last_valid_index()

        missing_days = int(series.isna().sum())
        missing_pct = missing_days / len(series) * 100

        if last_valid is None:
            stale_days = np.nan
        else:
            stale_days = (latest_market_date - pd.Timestamp(last_valid)).days

        rows.append({
            "ticker": ticker,
            "first_valid_date": first_valid,
            "last_valid_date": last_valid,
            "missing_days": missing_days,
            "missing_pct": round(missing_pct, 2),
            "stale_calendar_days": stale_days,
        })

    return (
        pd.DataFrame(rows)
        .set_index("ticker")
        .sort_values("missing_pct", ascending=False)
    )


def find_tickers_needing_repair(report):
    """Find empty, stale, or severely incomplete assets."""
    stale = pd.to_numeric(report["stale_calendar_days"], errors="coerce")

    repair_mask = (
        (report["missing_pct"] > MISSING_THRESHOLD_PCT)
        | stale.isna()
        | (stale > MAX_STALENESS_DAYS)
    )

    return report.index[repair_mask].tolist()


def validate_recent_coverage(close_prices, tickers):
    """Ensure every configured asset has recent observations."""
    latest_market_date = pd.Timestamp(close_prices.index.max())

    failures = []

    for ticker in tickers:
        if ticker not in close_prices.columns:
            failures.append(f"{ticker}: missing column")
            continue

        last_valid = close_prices[ticker].last_valid_index()

        if last_valid is None:
            failures.append(f"{ticker}: no observations")
            continue

        stale_days = (latest_market_date - pd.Timestamp(last_valid)).days

        if stale_days > MAX_STALENESS_DAYS:
            failures.append(
                f"{ticker}: last observation is {pd.Timestamp(last_valid).date()}"
            )

    if failures:
        formatted = "\n  - ".join(failures)

        raise RuntimeError(
            "Price coverage validation failed:\n"
            f"  - {formatted}\n\n"
            "Do not remove this validation. "
            "Wait for the Yahoo rate limit to clear "
            "and rerun the pipeline."
        )

    print(
        f"Coverage validation passed for "
        f"{len(tickers)} assets through "
        f"{latest_market_date.date()}."
    )


# ============================================================
# EUR conversion
# ============================================================

def download_fx_rate(pair, start, end):
    """Download one daily FX conversion series."""
    try:
        close, _ = download_prices([pair], start, end, progress=False)

    except Exception:
        close, _ = download_one_yahoo_history(ticker=pair, start=start, end=end)

    if pair not in close.columns or close[pair].dropna().empty:
        raise RuntimeError(f"No FX data returned for {pair}.")

    return close[pair].dropna()


def convert_to_eur(close_prices, non_eur_map):
    """Convert configured non-EUR prices into EUR."""
    converted = close_prices.copy()

    if not non_eur_map:
        print("No FX conversion is required.")
        return converted

    fx_rates = {}

    fx_start = pd.Timestamp(converted.index.min()).strftime("%Y-%m-%d")

    for pair in sorted(set(non_eur_map.values())):
        print(f"Downloading FX rate: {pair}")

        fx_rates[pair] = download_fx_rate(
            pair=pair,
            start=fx_start,
            end=DOWNLOAD_END_DATE,
        )

    for ticker, pair in non_eur_map.items():
        if ticker not in converted.columns:
            print(f"WARNING: {ticker} is absent from the price data.")
            continue

        rate = fx_rates[pair].reindex(converted.index).ffill()

        converted[ticker] = converted[ticker] * rate

        print(f"Converted {ticker} using {pair}.")

    return converted


# ============================================================
# Final cleaning
# ============================================================

def clean_prices(close_prices):
    """
    Forward-fill normal market-calendar gaps.

    Backward-fill is not used because it introduces future information.
    """
    return close_prices.sort_index().ffill().dropna(how="all")


def build_download_plan(existing_prices, configured_tickers):
    """
    Separate assets into:
      - full_history: absent or more than 20% missing
      - incremental: present but not updated close to END_DATE
    """
    configured_tickers = list(configured_tickers)

    if existing_prices.empty:
        return configured_tickers, []

    full_history = []
    incremental = []

    # Two calendar days before the exclusive end date.
    update_cutoff = pd.Timestamp(DOWNLOAD_END_DATE) - pd.Timedelta(days=2)

    for ticker in configured_tickers:
        if ticker not in existing_prices.columns:
            full_history.append(ticker)
            continue

        series = existing_prices[ticker]

        if not series.notna().any():
            full_history.append(ticker)
            continue

        coverage = series.notna().sum() / len(existing_prices)

        if coverage < 0.80:
            full_history.append(ticker)
            continue

        last_valid = pd.Timestamp(series.last_valid_index())

        if last_valid < update_cutoff:
            incremental.append(ticker)

    return full_history, incremental


def save_download_checkpoint(close_prices, volume, configured_tickers):
    """
    Save every successful download immediately.

    A later run can continue from these files without losing completed
    assets.
    """
    close_prices = restrict_to_requested_window(close_prices).reindex(
        columns=configured_tickers
    )

    volume = restrict_to_requested_window(volume).reindex(
        columns=configured_tickers
    )

    close_prices.to_csv(RAW_PRICES_FILE)
    volume.to_csv(RAW_VOLUME_FILE)

    print(
        f"Checkpoint saved: "
        f"{close_prices.notna().any().sum()}/"
        f"{len(configured_tickers)} assets contain prices."
    )

    return close_prices, volume

# ============================================================
# Main pipeline
# ============================================================

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    validate_date_configuration()

    print("\n" + "=" * 70)
    print("RESUMABLE PRICE-DATA PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Step 0: build the list of all entities from data_config.py
    # --------------------------------------------------------

    print("\nSTEP 0 — Building entity list from data_config.py")

    configured_tickers = list(ALL_TICKERS)

    print(f"Total entities in ALL_TICKERS: {len(configured_tickers)}")
    print(f"  ETFs:     {len(ETFS)}")
    print(f"  Equities: {len(EQUITIES)}")
    print("Entity list:")
    print(", ".join(configured_tickers))

    # --------------------------------------------------------
    # Step 1: load successful data from previous runs
    # --------------------------------------------------------

    print("\nSTEP 1 — Loading saved observations")

    close_prices = read_existing_file(RAW_PRICES_FILE)
    volume = read_existing_file(RAW_VOLUME_FILE)

    close_prices = close_prices.reindex(columns=configured_tickers)
    volume = volume.reindex(columns=configured_tickers)

    completed_before = int(close_prices.notna().any().sum())

    print(
        f"Previously saved assets: "
        f"{completed_before}/"
        f"{len(configured_tickers)}"
    )

    # --------------------------------------------------------
    # Step 2: initial download (recursive batches) or incremental update
    # --------------------------------------------------------

    print("\nSTEP 2 — Building download plan")

    full_history_tickers, incremental_tickers = build_download_plan(
        close_prices,
        configured_tickers,
    )

    # First-ever run: build the full ticker list and work through it in
    # recursive batches of BATCH_SIZE, dropping each ticker from the
    # pending queue as soon as it comes back with usable data.
    if close_prices.empty or completed_before == 0:
        print(
            f"No previous data found. Downloading all "
            f"{len(configured_tickers)} tickers in batches "
            f"of {BATCH_SIZE}."
        )

        close_prices, volume = download_batch_recursive(
            pending_tickers=configured_tickers,
            close_prices=close_prices,
            volume=volume,
            configured_tickers=configured_tickers,
        )

    # Existing assets only need the new dates appended.
    elif incremental_tickers:
        print(f"Assets needing a date update: {len(incremental_tickers)}")

        valid_last_dates = [
            close_prices[ticker].last_valid_index()
            for ticker in incremental_tickers
            if (
                ticker in close_prices.columns
                and close_prices[ticker].last_valid_index() is not None
            )
        ]

        if valid_last_dates:
            incremental_start = max(
                pd.Timestamp(DOWNLOAD_START_DATE),
                min(pd.Timestamp(date) for date in valid_last_dates)
                - pd.Timedelta(days=INCREMENTAL_OVERLAP_DAYS),
            ).strftime("%Y-%m-%d")
        else:
            incremental_start = DOWNLOAD_START_DATE

        try:
            fresh_close, fresh_volume = download_prices(
                tickers=incremental_tickers,
                start=incremental_start,
                end=DOWNLOAD_END_DATE,
            )

            # Fresh values win; saved history fills everything else.
            close_prices = fresh_close.combine_first(close_prices)
            volume = fresh_volume.combine_first(volume)

            close_prices, volume = save_download_checkpoint(
                close_prices,
                volume,
                configured_tickers,
            )

        except Exception as error:
            print(f"Incremental update failed: {str(error)[:200]}")
            print("Previously saved observations have been preserved.")

    else:
        print("Saved assets already reach the requested end date.")

    # --------------------------------------------------------
    # Step 3: identify missing or new assets again
    # --------------------------------------------------------

    full_history_tickers, _ = build_download_plan(close_prices, configured_tickers)

    print("\nSTEP 3 — Missing/new asset recovery")

    if full_history_tickers:
        print("Assets requiring full-history downloads:")
        print(", ".join(full_history_tickers))
    else:
        print("No full-history downloads are required.")

    # --------------------------------------------------------
    # Step 4: download missing assets one at a time
    # --------------------------------------------------------

    consecutive_rate_limit_hits = 0

    for position, ticker in enumerate(full_history_tickers, start=1):
        print(f"\n{position}/{len(full_history_tickers)} downloading {ticker}")

        try:
            one_close, one_volume = download_one_history_with_fallback(
                ticker=ticker,
                start=DOWNLOAD_START_DATE,
                end=DOWNLOAD_END_DATE,
            )

            # Append the newly recovered ticker while preserving all
            # previously saved tickers and dates.
            close_prices = one_close.combine_first(close_prices)
            volume = one_volume.combine_first(volume)

            # Save immediately after every successful asset.
            close_prices, volume = save_download_checkpoint(
                close_prices,
                volume,
                configured_tickers,
            )

            consecutive_rate_limit_hits = 0

        except Exception as error:
            print(f"Could not download {ticker}: {str(error)[:200]}")

            if "429" in str(error):
                consecutive_rate_limit_hits += 1

                if consecutive_rate_limit_hits >= MAX_CONSECUTIVE_RATE_LIMIT_HITS:
                    print(
                        f"\n{consecutive_rate_limit_hits} rate-limit hits in a "
                        f"row — the connection looks locked out. Stopping "
                        f"further requests."
                    )
                    print(
                        "All successful assets have already "
                        "been saved. Run the script later to "
                        "continue with the remaining assets."
                    )
                    break

                print(
                    f"Rate-limited on {ticker} "
                    f"({consecutive_rate_limit_hits}/{MAX_CONSECUTIVE_RATE_LIMIT_HITS} "
                    f"in a row) — skipping it for now and trying the next "
                    f"ticker instead of stopping the whole run."
                )

                time.sleep(RATE_LIMIT_BACKOFF_SECONDS)
                continue

        time.sleep(YAHOO_REQUEST_PAUSE_SECONDS)

    # --------------------------------------------------------
    # Step 5: final availability check
    # --------------------------------------------------------

    print("\nSTEP 5 — Checking completed assets")

    availability_report = report_missing(close_prices)

    unavailable = [
        ticker
        for ticker in configured_tickers
        if (
            ticker not in close_prices.columns
            or not close_prices[ticker].notna().any()
        )
    ]

    incomplete = availability_report[
        availability_report["missing_pct"] > MISSING_THRESHOLD_PCT
    ]

    incomplete.to_csv(BAD_TICKERS_FILE)

    completed_after = int(close_prices.notna().any().sum())

    print(
        f"Successfully stored: "
        f"{completed_after}/"
        f"{len(configured_tickers)} assets"
    )

    if unavailable:
        print("\nThe following assets are still unavailable:")
        print(", ".join(unavailable))
        print("\nPartial raw data has been saved safely.")
        print(
            "Run this same script later. It will preserve "
            "the completed assets and request only the "
            "missing assets."
        )

        # Do not generate the final clean model dataset while entire
        # asset columns remain empty.
        return

    # --------------------------------------------------------
    # Step 6: validate the complete native-price dataset
    # --------------------------------------------------------

    validate_recent_coverage(close_prices, configured_tickers)

    close_prices.to_csv(PATCHED_PRICES_FILE)

    # --------------------------------------------------------
    # Step 7: currency conversion
    # --------------------------------------------------------

    print("\nSTEP 7 — Converting prices to EUR")

    eur_prices = convert_to_eur(close_prices, NON_EUR_TICKERS)

    eur_prices = restrict_to_requested_window(eur_prices).reindex(
        columns=configured_tickers
    )

    eur_prices.to_csv(EUR_PRICES_FILE)

    # --------------------------------------------------------
    # Step 8: final cleaning
    # --------------------------------------------------------

    print("\nSTEP 8 — Cleaning final dataset")

    clean = clean_prices(eur_prices)

    clean.to_csv(CLEAN_PRICES_FILE)

    missing_after = int(clean.isna().sum().sum())

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    print(f"Final shape: {clean.shape[0]:,} dates x {clean.shape[1]} assets")
    print(f"Date range: {clean.index.min().date()} to {clean.index.max().date()}")
    print(f"Remaining missing values: {missing_after:,}")
    print(f"Final file: {CLEAN_PRICES_FILE}")


if __name__ == "__main__":
    main()