#!/usr/bin/env python3
"""Collect the broadest practical news set for the configured assets.

This is a drop-in replacement for the previous Yahoo-only backfill script.
It preserves the existing NEWS_FILE and adds in-window headlines from:

* Yahoo Finance (recent ticker-linked news)
* Marketaux (optional free API token, with pagination)
* GDELT DOC 2.0 (free/keyless global news search)
* Media Cloud (historical Online News Archive with resumable date windows)

The collector queries every requested ticker, not just tickers with zero rows.
It deduplicates the merged dataset and writes two audit files next to NEWS_FILE:
``news_coverage.csv`` and ``news_collection_failures.csv``. Use ``--check-only``
to inspect per-asset coverage without making any API requests.

Marketaux setup (optional):
    1. Keep .env next to data_config.py.
    2. Put this line in .env:
       MARKETAUX_API_TOKEN=your_new_api_key
    3. data_config.py loads it; do not paste the key into this script.

Media Cloud setup (optional):
    1. Install the official client: python -m pip install mediacloud
    2. Put MEDIACLOUD_API_KEY=your_key in the same .env file.
    3. The script discovers suitable European national news collections. If
       discovery is unavailable, supply IDs with --media-cloud-collection-ids.
    4. Re-run the same command after a request-budget stop; completed archive
       windows are skipped automatically.

Examples:
    python news_collection.py
    python news_collection.py --tickers SAP.DE ASML.AS NESN.SW
    python news_collection.py --providers yahoo,gdelt
    python news_collection.py --providers media_cloud --tickers RO.SW

Important: no free news service is a complete licensed historical archive.
Yahoo is recent-only; GDELT's DOC API is primarily a rolling recent window;
Marketaux's free tier has a daily request/article limit; and Media Cloud is a
research archive rather than a licensed exchange feed. This script
downloads all results exposed by the enabled services within those limits and
reports any truncation, provider failure, or remaining coverage gap.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import sys
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yf

import data_config

from data_config import (
    ALL_TICKERS,
    DATA_DIR,
    END_DATE,
    MARKETAUX_API_TOKEN,
    NEWS_FILE,
    START_DATE,
)


NEWS_COLUMNS = ["ticker", "headline", "publish_time", "source"]
FAILURE_COLUMNS = ["ticker", "provider", "error"]
MEDIA_CLOUD_RAW_COLUMNS = NEWS_COLUMNS + [
    "publisher",
    "url",
    "query",
    "collection_ids",
    "window_start",
    "window_end",
]
SUPPORTED_PROVIDERS = {"yahoo", "marketaux", "gdelt", "media_cloud"}

YAHOO_NEWS_COUNT = 100
REQUEST_RETRIES = 3
PAUSE_BETWEEN_TICKERS_SECONDS = 0.25

MARKETAUX_URL = "https://api.marketaux.com/v1/news/all"
MARKETAUX_FREE_PAGE_SIZE = 3
DEFAULT_MARKETAUX_DAILY_BUDGET = 95
DEFAULT_MARKETAUX_MAX_PAGES = 2

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_MAX_RECORDS = 250
GDELT_SECONDS_BETWEEN_CALLS = 5.1
GDELT_RECENT_WINDOW_DAYS = 90
GDELT_MIN_SPLIT_HOURS = 24
DEFAULT_GDELT_MAX_CALLS = 500

DEFAULT_MEDIA_CLOUD_REQUEST_BUDGET = 1_000
DEFAULT_MEDIA_CLOUD_MAX_PAGES_PER_WINDOW = 200
DEFAULT_MEDIA_CLOUD_WINDOW_MONTHS = 12
DEFAULT_MEDIA_CLOUD_PAUSE_SECONDS = 0.10
MEDIA_CLOUD_SOURCE = "media_cloud"
MEDIA_CLOUD_TARGET_COUNTRIES = {
    "germany": ("germany", "german"),
    "france": ("france", "french"),
    "italy": ("italy", "italian"),
    "spain": ("spain", "spanish"),
    "netherlands": ("netherlands", "dutch"),
    "switzerland": ("switzerland", "swiss"),
}
MEDIA_CLOUD_COUNTRY_COLLECTION_NAMES = {
    "germany": "Germany - National",
    "france": "France - National",
    "italy": "Italy - National",
    "spain": "Spain - National",
    "netherlands": "Netherlands - National",
    "switzerland": "Switzerland - National",
}
MEDIA_CLOUD_TICKER_SUFFIX_COUNTRIES = {
    ".DE": "germany",
    ".PA": "france",
    ".MI": "italy",
    ".MC": "spain",
    ".AS": "netherlands",
    ".SW": "switzerland",
}
MEDIA_CLOUD_COUNTRY_ETFS = {
    "EWI": "italy",
    "EWP": "spain",
    "EWN": "netherlands",
    "EWL": "switzerland",
}
MEDIA_CLOUD_BROAD_ETF_PREFIXES = ("EXSA", "EXV", "EXH")
DEFAULT_COVERAGE_START_YEAR = 2019
DEFAULT_COVERAGE_END_YEAR = 2025

EXTEND_END_DATE_TO_TODAY = False

_SSL_CONTEXT: ssl.SSLContext | None = None


@dataclass
class RequestBudget:
    limit: int
    used: int = 0

    def available(self) -> bool:
        return self.used < self.limit

    def consume(self) -> None:
        if not self.available():
            raise RuntimeError(f"request budget exhausted ({self.limit} calls)")
        self.used += 1


class RateLimiter:
    def __init__(self, minimum_interval: float) -> None:
        self.minimum_interval = minimum_interval
        self.last_call = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self.last_call
        if self.last_call and elapsed < self.minimum_interval:
            time.sleep(self.minimum_interval - elapsed)
        self.last_call = time.monotonic()


def get_date_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return UTC start and exclusive end timestamps."""
    start_ts = pd.to_datetime(START_DATE, errors="raise", utc=True).normalize()
    configured_end = pd.to_datetime(END_DATE, errors="raise", utc=True).normalize()
    if configured_end < start_ts:
        raise ValueError("END_DATE must not be earlier than START_DATE")

    end_day = configured_end
    if EXTEND_END_DATE_TO_TODAY:
        end_day = max(end_day, pd.Timestamp.now(tz="UTC").normalize())
    return start_ts, end_day + pd.Timedelta(days=1)


def get_ssl_context() -> ssl.SSLContext:
    """Return a verified TLS context using the native OS trust store when possible.

    ``truststore`` handles OS-managed roots and missing intermediates on Python
    3.10+. Certifi is the verified fallback. Verification is never disabled.
    """
    global _SSL_CONTEXT
    if _SSL_CONTEXT is not None:
        return _SSL_CONTEXT

    custom_ca_file = os.getenv("SSL_CERT_FILE", "").strip()
    if custom_ca_file:
        if not os.path.isfile(custom_ca_file):
            raise RuntimeError(f"SSL_CERT_FILE does not exist: {custom_ca_file}")
        context = ssl.create_default_context(cafile=custom_ca_file)
        _SSL_CONTEXT = context
        return context

    try:
        import truststore
    except ImportError:
        context = ssl.create_default_context()
        try:
            import certifi
        except ImportError:
            pass
        else:
            context.load_verify_locations(cafile=certifi.where())
    else:
        context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
    _SSL_CONTEXT = context
    return context


def parse_timestamp(value: Any) -> pd.Timestamp:
    if value is None or value == "":
        return pd.NaT
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return pd.NaT
        if value.isdigit():
            value = int(value)
    if isinstance(value, (int, float)):
        unit = "ms" if abs(value) >= 1_000_000_000_000 else "s"
        return pd.to_datetime(value, unit=unit, errors="coerce", utc=True)

    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed) and isinstance(value, str):
        parsed = pd.to_datetime(
            value, format="%Y%m%dT%H%M%SZ", errors="coerce", utc=True
        )
    return parsed


def in_window(
    publish_time: pd.Timestamp,
    start_ts: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> bool:
    return pd.notna(publish_time) and start_ts <= publish_time < end_exclusive


def load_existing_news(news_file: str | os.PathLike[str] = NEWS_FILE) -> pd.DataFrame:
    path = Path(news_file)
    if not path.exists():
        print(f"No existing news file found at {path}; creating it")
        return pd.DataFrame(columns=NEWS_COLUMNS)

    news_df = pd.read_csv(path)
    missing_columns = set(NEWS_COLUMNS) - set(news_df.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {sorted(missing_columns)}")

    news_df = news_df[NEWS_COLUMNS].copy()
    news_df["ticker"] = news_df["ticker"].astype("string").str.strip().str.upper()
    news_df["headline"] = news_df["headline"].astype("string").str.strip()
    news_df["source"] = news_df["source"].astype("string").str.strip().str.lower()
    news_df["publish_time"] = pd.to_datetime(
        news_df["publish_time"], errors="coerce", utc=True
    )
    print(f"Loaded {len(news_df):,} existing headlines from {path}")
    return news_df


def request_json(
    url: str,
    params: dict[str, Any],
    provider: str,
    rate_limiter: RateLimiter | None = None,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            if rate_limiter:
                rate_limiter.wait()
            request = Request(
                f"{url}?{urlencode(params)}",
                headers={
                    "Accept": "application/json",
                    "User-Agent": "europe-news-collector/2.0",
                },
            )
            with urlopen(request, timeout=45, context=get_ssl_context()) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
            if not isinstance(payload, dict):
                raise ValueError("response was not a JSON object")
            return payload
        except HTTPError as exc:
            last_error = exc
            if exc.code == 429:
                retry_after = exc.headers.get("Retry-After", "5")
                try:
                    wait_seconds = float(retry_after)
                except ValueError:
                    wait_seconds = 5.0
                if attempt < REQUEST_RETRIES:
                    time.sleep(max(wait_seconds, 5.1))
                    continue
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
        except (URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            if isinstance(exc, URLError) and isinstance(
                exc.reason, ssl.SSLCertVerificationError
            ):
                last_error = RuntimeError(
                    "TLS certificate verification failed. Run 'python -m pip "
                    "install --upgrade truststore certifi'. If your network uses "
                    "a corporate proxy, ask IT for its approved CA bundle and set "
                    "SSL_CERT_FILE to that file. Verification was not disabled."
                )
            else:
                last_error = exc
        if attempt < REQUEST_RETRIES:
            time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"{provider} failed after {REQUEST_RETRIES} attempts: {last_error}")


def unpack_yahoo_item(item: Any) -> tuple[str, pd.Timestamp]:
    if not isinstance(item, dict):
        return "", pd.NaT
    nested = item.get("content")
    content = nested if isinstance(nested, dict) else item
    headline = str(content.get("title") or item.get("title") or "").strip()
    raw_time = (
        content.get("providerPublishTime")
        or content.get("pubDate")
        or content.get("displayTime")
        or item.get("providerPublishTime")
        or item.get("pubDate")
    )
    return headline, parse_timestamp(raw_time)


def fetch_yahoo_news(
    ticker: str,
    start_ts: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> list[dict[str, Any]]:
    last_error: Exception | None = None
    items: list[Any] = []
    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            stock = yf.Ticker(ticker)
            try:
                items = stock.get_news(count=YAHOO_NEWS_COUNT, tab="news") or []
            except (AttributeError, TypeError):
                items = stock.news or []
            break
        except Exception as exc:  # yfinance exposes several transport errors.
            last_error = exc
            if attempt < REQUEST_RETRIES:
                time.sleep(2 ** (attempt - 1))
    else:
        raise RuntimeError(f"Yahoo failed after {REQUEST_RETRIES} attempts: {last_error}")

    rows = []
    for item in items:
        headline, publish_time = unpack_yahoo_item(item)
        if headline and in_window(publish_time, start_ts, end_exclusive):
            rows.append(
                {
                    "ticker": ticker,
                    "headline": headline,
                    "publish_time": publish_time,
                    "source": "yahoo_finance",
                }
            )
    return rows


def marketaux_params(
    ticker: str,
    query_name: str,
    token: str,
    start_ts: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    page: int,
    use_symbol: bool,
) -> dict[str, Any]:
    # Marketaux accepts date/time values only through whole seconds and without
    # a timezone suffix (for example, 2026-01-01T12:30:00). Pandas isoformat()
    # emits "+00:00" and, for the old end value, fractional seconds; Marketaux
    # rejects both as malformed. Expand the server-side bounds by one second,
    # then retain the exact configured window in parse_marketaux_rows().
    published_after = (start_ts - pd.Timedelta(seconds=1)).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    published_before = end_exclusive.strftime("%Y-%m-%dT%H:%M:%S")

    params: dict[str, Any] = {
        "api_token": token,
        "filter_entities": "true",
        "language": "en,de,fr,it,es,nl",
        "published_after": published_after,
        "published_before": published_before,
        "sort": "published_desc",
        "limit": MARKETAUX_FREE_PAGE_SIZE,
        "page": page,
    }
    if use_symbol:
        params["symbols"] = ticker
    else:
        params["search"] = query_name
    return params


def parse_marketaux_rows(
    ticker: str,
    payload: dict[str, Any],
    start_ts: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> list[dict[str, Any]]:
    rows = []
    for item in payload.get("data") or []:
        if not isinstance(item, dict):
            continue
        headline = str(item.get("title") or "").strip()
        publish_time = parse_timestamp(item.get("published_at"))
        if headline and in_window(publish_time, start_ts, end_exclusive):
            rows.append(
                {
                    "ticker": ticker,
                    "headline": headline,
                    "publish_time": publish_time,
                    "source": "marketaux",
                }
            )
    return rows


def fetch_marketaux_news(
    ticker: str,
    query_name: str,
    token: str,
    start_ts: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    budget: RequestBudget,
    max_pages: int,
) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    use_symbol = True

    for page in range(1, max_pages + 1):
        budget.consume()
        payload = request_json(
            MARKETAUX_URL,
            marketaux_params(
                ticker, query_name, token, start_ts, end_exclusive, page, use_symbol
            ),
            "Marketaux",
        )
        raw_rows = payload.get("data") or []

        # Some European/ETF symbols are not recognized. Spend one call on the
        # unambiguous company/fund name before declaring the asset uncovered.
        if page == 1 and not raw_rows:
            budget.consume()
            use_symbol = False
            payload = request_json(
                MARKETAUX_URL,
                marketaux_params(
                    ticker,
                    query_name,
                    token,
                    start_ts,
                    end_exclusive,
                    page,
                    use_symbol,
                ),
                "Marketaux",
            )
            raw_rows = payload.get("data") or []

        all_rows.extend(parse_marketaux_rows(ticker, payload, start_ts, end_exclusive))
        if len(raw_rows) < MARKETAUX_FREE_PAGE_SIZE:
            break

        meta = payload.get("meta") or {}
        found = meta.get("found")
        if isinstance(found, int) and page * MARKETAUX_FREE_PAGE_SIZE >= found:
            break

    return all_rows


def gdelt_time(timestamp: pd.Timestamp) -> str:
    return timestamp.tz_convert("UTC").strftime("%Y%m%d%H%M%S")


def parse_gdelt_rows(
    ticker: str,
    payload: dict[str, Any],
    start_ts: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> list[dict[str, Any]]:
    rows = []
    for item in payload.get("articles") or []:
        if not isinstance(item, dict):
            continue
        headline = str(item.get("title") or "").strip()
        publish_time = parse_timestamp(item.get("seendate"))
        if headline and in_window(publish_time, start_ts, end_exclusive):
            rows.append(
                {
                    "ticker": ticker,
                    "headline": headline,
                    "publish_time": publish_time,
                    "source": "gdelt",
                }
            )
    return rows


def fetch_gdelt_interval(
    ticker: str,
    query_name: str,
    start_ts: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    budget: RequestBudget,
    rate_limiter: RateLimiter,
) -> list[dict[str, Any]]:
    """Fetch an interval and bisect saturated result sets to avoid truncation."""
    budget.consume()
    payload = request_json(
        GDELT_URL,
        {
            "query": f'"{query_name}"',
            "mode": "artlist",
            "format": "json",
            "sort": "datedesc",
            "maxrecords": GDELT_MAX_RECORDS,
            "startdatetime": gdelt_time(start_ts),
            "enddatetime": gdelt_time(end_exclusive - pd.Timedelta(seconds=1)),
        },
        "GDELT",
        rate_limiter,
    )
    raw_rows = payload.get("articles") or []
    interval = end_exclusive - start_ts

    if len(raw_rows) >= GDELT_MAX_RECORDS and interval > pd.Timedelta(
        hours=GDELT_MIN_SPLIT_HOURS
    ):
        midpoint = start_ts + interval / 2
        left = fetch_gdelt_interval(
            ticker,
            query_name,
            start_ts,
            midpoint,
            budget,
            rate_limiter,
        )
        right = fetch_gdelt_interval(
            ticker,
            query_name,
            midpoint,
            end_exclusive,
            budget,
            rate_limiter,
        )
        return left + right

    if len(raw_rows) >= GDELT_MAX_RECORDS:
        print(
            f"  WARNING: GDELT still returned {GDELT_MAX_RECORDS} rows for a "
            f"minimum-size interval; {ticker} may be truncated",
            file=sys.stderr,
        )
    return parse_gdelt_rows(ticker, payload, start_ts, end_exclusive)


def fetch_gdelt_news(
    ticker: str,
    query_name: str,
    start_ts: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    budget: RequestBudget,
    rate_limiter: RateLimiter,
) -> list[dict[str, Any]]:
    now = pd.Timestamp.now(tz="UTC")
    recent_floor = now - pd.Timedelta(days=GDELT_RECENT_WINDOW_DAYS)
    effective_start = max(start_ts, recent_floor)
    effective_end = min(end_exclusive, now + pd.Timedelta(seconds=1))
    if effective_start >= effective_end:
        return []
    return fetch_gdelt_interval(
        ticker,
        query_name,
        effective_start,
        effective_end,
        budget,
        rate_limiter,
    )


@dataclass
class MediaCloudState:
    client: Any
    budget: RequestBudget
    collection_ids: list[int]
    collection_names: list[str]
    windows: list[tuple[date, date]]
    completed: set[str]
    raw_frame: pd.DataFrame
    raw_path: Path
    progress_path: Path
    max_pages_per_window: int
    pause_seconds: float


def add_months(day: date, months: int) -> date:
    """Move a date forward by whole months without crossing month length."""
    month_index = day.year * 12 + day.month - 1 + months
    year, zero_based_month = divmod(month_index, 12)
    month = zero_based_month + 1
    leap_year = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    month_lengths = [
        31,
        29 if leap_year else 28,
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ]
    return date(year, month, min(day.day, month_lengths[month - 1]))


def make_media_cloud_windows(
    start_ts: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    months: int,
) -> list[tuple[date, date]]:
    """Return inclusive Media Cloud date windows without gaps."""
    windows: list[tuple[date, date]] = []
    current = start_ts.date()
    final = (end_exclusive - pd.Timedelta(days=1)).date()
    while current <= final:
        next_start = add_months(current, months)
        window_end = min(final, next_start - timedelta(days=1))
        windows.append((current, window_end))
        current = window_end + timedelta(days=1)
    return windows


def media_cloud_paths() -> tuple[Path, Path]:
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    return (
        data_dir / "media_cloud_news_raw.csv",
        data_dir / "media_cloud_progress.json",
    )


def load_csv_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[columns].copy()


def load_media_cloud_progress(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    completed = payload.get("completed_windows", [])
    if not isinstance(completed, list):
        raise ValueError(f"Invalid Media Cloud progress file: {path}")
    return {str(item) for item in completed}


def media_cloud_progress_key(
    ticker: str,
    window_start: date,
    window_end: date,
    collection_ids: Iterable[int],
) -> str:
    collection_signature = ",".join(str(value) for value in sorted(collection_ids))
    return (
        f"{ticker}|{window_start.isoformat()}|{window_end.isoformat()}"
        f"|collections={collection_signature}"
    )


def parse_media_cloud_collection_ids(raw_value: Any) -> list[int]:
    """Normalize CLI, config, or .env collection IDs into positive integers."""
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        values: Iterable[Any] = re.split(r"[\s,;]+", raw_value.strip())
    elif isinstance(raw_value, Iterable):
        values = raw_value
    else:
        values = [raw_value]

    result: list[int] = []
    for value in values:
        if value in (None, ""):
            continue
        try:
            collection_id = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid Media Cloud collection ID: {value!r}"
            ) from exc
        if collection_id <= 0:
            raise ValueError("Media Cloud collection IDs must be positive")
        if collection_id not in result:
            result.append(collection_id)
    placeholder_ids = {123456, 789012}
    if placeholder_ids.intersection(result):
        raise ValueError(
            "123456 and 789012 are example placeholders, not Media Cloud "
            "collection IDs. Remove MEDIACLOUD_COLLECTION_IDS from .env so "
            "the script can discover real collections."
        )
    return result


def media_cloud_collection_records(response: Any) -> list[dict[str, Any]]:
    """Extract directory collection records across client response versions."""
    if isinstance(response, dict):
        records = response.get("results", [])
    else:
        records = response
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict)]


def select_european_media_cloud_collections(
    records: Iterable[dict[str, Any]],
) -> tuple[list[int], list[str]]:
    """Select only the six exact national collections used by this universe."""
    expected_names = {
        value.casefold() for value in MEDIA_CLOUD_COUNTRY_COLLECTION_NAMES.values()
    }
    selected: list[tuple[int, str]] = []
    seen: set[int] = set()
    for record in records:
        name = str(record.get("name") or record.get("label") or "").strip()
        normalized_name = name.casefold()
        if normalized_name not in expected_names:
            continue
        raw_id = (
            record.get("id")
            or record.get("collection_id")
            or record.get("collections_id")
        )
        try:
            collection_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if collection_id > 0 and collection_id not in seen:
            selected.append((collection_id, name or str(collection_id)))
            seen.add(collection_id)
    selected.sort(key=lambda item: item[1].casefold())
    return [item[0] for item in selected], [item[1] for item in selected]


def media_cloud_ticker_country(ticker: str) -> str | None:
    """Return the asset's home country, or None for broad European ETFs."""
    normalized = ticker.strip().upper()
    if normalized.startswith(MEDIA_CLOUD_BROAD_ETF_PREFIXES):
        return None
    if normalized in MEDIA_CLOUD_COUNTRY_ETFS:
        return MEDIA_CLOUD_COUNTRY_ETFS[normalized]
    for suffix, country in MEDIA_CLOUD_TICKER_SUFFIX_COUNTRIES.items():
        if normalized.endswith(suffix):
            return country
    return None


def media_cloud_collection_ids_for_ticker(
    state: MediaCloudState,
    ticker: str,
) -> list[int]:
    """Use one national collection for an equity and all six for broad ETFs."""
    country = media_cloud_ticker_country(ticker)
    if country is None:
        return list(state.collection_ids)
    expected_name = MEDIA_CLOUD_COUNTRY_COLLECTION_NAMES[country].casefold()
    matched = [
        collection_id
        for collection_id, name in zip(
            state.collection_ids,
            state.collection_names,
        )
        if name.casefold() == expected_name
    ]
    # Explicit CLI/config IDs have generic names. In that case the caller
    # explicitly chose the scope, so preserve every supplied ID.
    return matched or list(state.collection_ids)


def resolve_media_cloud_collections(
    api_module: Any,
    api_key: str,
    budget: RequestBudget,
    explicit_ids: Iterable[int] | None,
) -> tuple[list[int], list[str]]:
    """Use explicit IDs or discover European national collections once."""
    collection_ids = parse_media_cloud_collection_ids(explicit_ids)
    if collection_ids:
        return collection_ids, [f"collection {value}" for value in collection_ids]

    configured_ids = parse_media_cloud_collection_ids(
        getattr(data_config, "MEDIACLOUD_COLLECTION_IDS", None)
        or os.getenv("MEDIACLOUD_COLLECTION_IDS", "")
    )
    if configured_ids:
        return configured_ids, [f"collection {value}" for value in configured_ids]

    if not budget.available():
        raise RuntimeError(
            "Media Cloud request budget is too small for collection discovery"
        )

    directory = api_module.DirectoryApi(api_key)
    platform = getattr(directory, "PLATFORM_ONLINE_NEWS", "online_news")
    budget.consume()
    response = directory.collection_list(
        platform=platform,
        name="National",
        limit=1000,
        offset=0,
    )
    records = media_cloud_collection_records(response)
    collection_ids, collection_names = select_european_media_cloud_collections(
        records
    )
    if not collection_ids:
        raise RuntimeError(
            "Media Cloud collection discovery returned no European national "
            "collections. Open the Media Cloud Directory, choose one or more "
            "collections, then rerun with --media-cloud-collection-ids ID ... "
            "or set MEDIACLOUD_COLLECTION_IDS=ID,ID in .env."
        )
    return collection_ids, collection_names


def media_cloud_aliases(ticker: str, configured_name: str) -> list[str]:
    raw_mapping = (
        getattr(data_config, "NEWS_SEARCH_ALIASES", {})
        or getattr(data_config, "ETF_NEWS_ALIASES", {})
        or {}
    )
    mapping = {str(key).strip().upper(): value for key, value in raw_mapping.items()}
    raw_aliases = mapping.get(ticker, [configured_name])
    if isinstance(raw_aliases, str):
        raw_aliases = [raw_aliases]

    aliases: list[str] = []
    seen: set[str] = set()
    for value in raw_aliases:
        alias = str(value).strip()
        key = alias.casefold()
        if alias and key not in seen:
            aliases.append(alias)
            seen.add(key)
    if not aliases:
        raise ValueError(f"No usable Media Cloud search alias for {ticker}")
    return aliases


def media_cloud_query(aliases: Iterable[str]) -> str:
    quoted = []
    for value in aliases:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        quoted.append(f'"{escaped}"')
    return " OR ".join(quoted)


def media_cloud_story_row(
    ticker: str,
    query: str,
    collection_ids: Iterable[int],
    window_start: date,
    window_end: date,
    story: dict[str, Any],
) -> dict[str, Any] | None:
    headline = str(story.get("title") or "").strip()
    publish_time = parse_timestamp(story.get("publish_date"))
    if not headline or pd.isna(publish_time):
        return None
    return {
        "ticker": ticker,
        "headline": headline,
        "publish_time": publish_time,
        "source": MEDIA_CLOUD_SOURCE,
        "publisher": str(story.get("media_name") or "").strip(),
        "url": str(story.get("url") or "").strip(),
        "query": query,
        "collection_ids": ",".join(str(value) for value in collection_ids),
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
    }


def fetch_media_cloud_window(
    state: MediaCloudState,
    ticker: str,
    query: str,
    collection_ids: list[int],
    window_start: date,
    window_end: date,
) -> tuple[list[dict[str, Any]], str | None]:
    """Fetch one window; return rows and an optional truncation reason."""
    rows: list[dict[str, Any]] = []
    pagination_token: str | None = None

    for _page_number in range(1, state.max_pages_per_window + 1):
        if not state.budget.available():
            return rows, "request budget reached before the window finished"
        state.budget.consume()
        page, next_token = state.client.story_list(
            query,
            start_date=window_start,
            end_date=window_end,
            collection_ids=collection_ids,
            pagination_token=pagination_token,
        )
        for story in page or []:
            if not isinstance(story, dict):
                continue
            row = media_cloud_story_row(
                ticker,
                query,
                collection_ids,
                window_start,
                window_end,
                story,
            )
            if row is not None:
                rows.append(row)
        print(
            f"    {window_start} to {window_end}: page {_page_number}, "
            f"{len(rows):,} headlines received",
            flush=True,
        )
        pagination_token = next_token
        if not pagination_token:
            return rows, None
        if state.pause_seconds:
            time.sleep(state.pause_seconds)

    return (
        rows,
        f"truncated at {state.max_pages_per_window} pages; reduce "
        "--media-cloud-window-months or increase --media-cloud-max-pages",
    )


def deduplicate_media_cloud_raw(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=MEDIA_CLOUD_RAW_COLUMNS)
    result = frame[MEDIA_CLOUD_RAW_COLUMNS].copy()
    result["ticker"] = result["ticker"].astype(str).str.strip().str.upper()
    result["headline"] = result["headline"].astype(str).str.strip()
    result["publish_time"] = pd.to_datetime(
        result["publish_time"], errors="coerce", utc=True
    )
    result = result.dropna(subset=["ticker", "headline", "publish_time"])
    result = result.loc[result["headline"].ne("")].copy()
    result["_headline_key"] = normalize_headline(result["headline"])
    result["_publish_day"] = result["publish_time"].dt.date
    result = result.drop_duplicates(
        subset=["ticker", "_headline_key", "_publish_day"], keep="first"
    )
    result = result.drop(columns=["_headline_key", "_publish_day"])
    return result.sort_values(
        ["ticker", "publish_time"], kind="stable"
    )[MEDIA_CLOUD_RAW_COLUMNS].reset_index(drop=True)


def save_media_cloud_checkpoint(state: MediaCloudState) -> None:
    state.raw_frame = deduplicate_media_cloud_raw(state.raw_frame)
    atomic_write_csv(state.raw_frame, state.raw_path)
    temporary = state.progress_path.with_suffix(state.progress_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "completed_windows": sorted(state.completed),
                "updated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, state.progress_path)


def create_media_cloud_state(
    start_ts: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    budget_limit: int,
    window_months: int,
    max_pages_per_window: int,
    pause_seconds: float,
    restart: bool,
    collection_ids: Iterable[int] | None,
) -> MediaCloudState:
    api_key = str(getattr(data_config, "MEDIACLOUD_API_KEY", "") or "").strip()
    if not api_key:
        api_key = os.getenv("MEDIACLOUD_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "MEDIACLOUD_API_KEY is empty. Add it to .env next to data_config.py."
        )

    try:
        import truststore
    except ImportError:
        pass
    else:
        try:
            truststore.inject_into_ssl()
        except Exception:
            # urllib requests still use get_ssl_context(); the Media Cloud
            # client will retain its own verified default context.
            pass

    try:
        import mediacloud.api
    except ImportError as exc:
        raise RuntimeError(
            "The mediacloud package is missing. Run: "
            "python -m pip install mediacloud"
        ) from exc

    budget = RequestBudget(budget_limit)
    resolved_ids, resolved_names = resolve_media_cloud_collections(
        mediacloud.api,
        api_key,
        budget,
        collection_ids,
    )
    print(
        "Media Cloud collections: "
        + "; ".join(
            f"{name} [{collection_id}]"
            for collection_id, name in zip(resolved_ids, resolved_names)
        )
    )

    raw_path, progress_path = media_cloud_paths()
    return MediaCloudState(
        client=mediacloud.api.SearchApi(api_key),
        budget=budget,
        collection_ids=resolved_ids,
        collection_names=resolved_names,
        windows=make_media_cloud_windows(start_ts, end_exclusive, window_months),
        completed=set() if restart else load_media_cloud_progress(progress_path),
        raw_frame=load_csv_columns(raw_path, MEDIA_CLOUD_RAW_COLUMNS),
        raw_path=raw_path,
        progress_path=progress_path,
        max_pages_per_window=max_pages_per_window,
        pause_seconds=pause_seconds,
    )


def collect_media_cloud_ticker(
    state: MediaCloudState,
    ticker: str,
    configured_name: str,
    windows: list[tuple[date, date]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], bool, int]:
    """Collect unfinished archive windows for one ticker and checkpoint each."""
    query = media_cloud_query(media_cloud_aliases(ticker, configured_name))
    ticker_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    processed_windows = 0
    exhausted = False
    ticker_windows = state.windows if windows is None else windows
    collection_ids = media_cloud_collection_ids_for_ticker(state, ticker)

    if not ticker_windows:
        return ticker_rows, failures, exhausted, processed_windows

    print(
        "  Media Cloud collection IDs: "
        + ",".join(str(value) for value in collection_ids),
        flush=True,
    )

    for window_start, window_end in ticker_windows:
        key = media_cloud_progress_key(
            ticker,
            window_start,
            window_end,
            collection_ids,
        )
        if key in state.completed:
            continue
        if not state.budget.available():
            exhausted = True
            break

        try:
            rows, truncation = fetch_media_cloud_window(
                state,
                ticker,
                query,
                collection_ids,
                window_start,
                window_end,
            )
        except Exception as exc:  # Official client raises provider-specific errors.
            error_text = f"{type(exc).__name__}: {exc}"
            failures.append(
                {
                    "ticker": ticker,
                    "provider": MEDIA_CLOUD_SOURCE,
                    "error": (
                        f"{window_start} to {window_end}: "
                        f"{error_text}"
                    ),
                }
            )
            print(
                f"    Media Cloud error for {window_start} to {window_end}: "
                f"{error_text}",
                file=sys.stderr,
                flush=True,
            )
            fatal_markers = (
                "401",
                "403",
                "unauthorized",
                "forbidden",
                "invalid api",
                "api key",
            )
            if any(marker in error_text.casefold() for marker in fatal_markers):
                exhausted = True
                break
            continue

        if rows:
            new_rows = pd.DataFrame(rows, columns=MEDIA_CLOUD_RAW_COLUMNS)
            if state.raw_frame.empty:
                state.raw_frame = new_rows
            else:
                state.raw_frame = pd.concat(
                    [state.raw_frame, new_rows],
                    ignore_index=True,
                )
            ticker_rows.extend(rows)

        if truncation is None:
            state.completed.add(key)
        else:
            failures.append(
                {
                    "ticker": ticker,
                    "provider": MEDIA_CLOUD_SOURCE,
                    "error": f"{window_start} to {window_end}: {truncation}",
                }
            )
            if "request budget reached" in truncation:
                exhausted = True

        processed_windows += 1
        save_media_cloud_checkpoint(state)
        if exhausted:
            break

    return ticker_rows, failures, exhausted, processed_windows


def recover_media_cloud_rows() -> pd.DataFrame:
    """Load checkpointed Media Cloud rows so interrupted runs merge on resume."""
    raw_path, _ = media_cloud_paths()
    raw = load_csv_columns(raw_path, MEDIA_CLOUD_RAW_COLUMNS)
    if raw.empty:
        return pd.DataFrame(columns=NEWS_COLUMNS)
    raw = deduplicate_media_cloud_raw(raw)
    return raw[NEWS_COLUMNS].copy()


def normalize_headline(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.casefold()
        .str.replace(r"[^\w]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def merge_news(
    existing_news: pd.DataFrame,
    downloaded_news: pd.DataFrame,
) -> pd.DataFrame:
    combined = pd.concat([existing_news, downloaded_news], ignore_index=True)
    combined["publish_time"] = pd.to_datetime(
        combined["publish_time"], errors="coerce", utc=True
    )
    combined = combined.dropna(subset=["ticker", "headline", "publish_time"])
    combined["ticker"] = combined["ticker"].astype(str).str.strip().str.upper()
    combined["headline"] = combined["headline"].astype(str).str.strip()
    combined["source"] = combined["source"].astype(str).str.strip().str.lower()
    combined = combined.loc[combined["headline"].ne("")].copy()

    combined["_headline_key"] = normalize_headline(combined["headline"])
    combined["_publish_day"] = combined["publish_time"].dt.date
    combined["_source_priority"] = (
        combined["source"]
        .map(
            {
                "yahoo_finance": 0,
                "marketaux": 1,
                "gdelt": 2,
                "media_cloud": 3,
            }
        )
        .fillna(5)
    )
    combined = combined.sort_values(
        ["_source_priority", "publish_time"], ascending=[True, False], kind="stable"
    )
    combined = combined.drop_duplicates(
        subset=["ticker", "_headline_key", "_publish_day"], keep="first"
    )
    combined = combined.drop(
        columns=["_headline_key", "_publish_day", "_source_priority"]
    )
    combined = combined.sort_values(
        ["ticker", "publish_time"], kind="stable"
    ).reset_index(drop=True)
    return combined[NEWS_COLUMNS]


def atomic_write_csv(dataframe: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    dataframe.to_csv(temporary, index=False)
    os.replace(temporary, destination)


def build_missing_year_windows(
    news_df: pd.DataFrame,
    tickers: Iterable[str],
    start_year: int,
    end_year: int,
) -> dict[str, list[tuple[date, date]]]:
    """Return minimal ranges, merging consecutive zero-headline years."""
    scoped = news_df.loc[news_df["ticker"].isin(tickers)].copy()
    scoped["year"] = pd.to_datetime(
        scoped["publish_time"], errors="coerce", utc=True
    ).dt.year
    covered = {
        (str(ticker), int(year))
        for ticker, year in scoped[["ticker", "year"]].dropna().itertuples(index=False)
    }
    windows_by_ticker: dict[str, list[tuple[date, date]]] = {}
    for ticker in tickers:
        missing_years = [
            year
            for year in range(start_year, end_year + 1)
            if (ticker, year) not in covered
        ]
        ranges: list[tuple[date, date]] = []
        if missing_years:
            range_start = missing_years[0]
            range_end = missing_years[0]
            for year in missing_years[1:]:
                if year == range_end + 1:
                    range_end = year
                    continue
                ranges.append(
                    (date(range_start, 1, 1), date(range_end, 12, 31))
                )
                range_start = year
                range_end = year
            ranges.append((date(range_start, 1, 1), date(range_end, 12, 31)))
        windows_by_ticker[ticker] = ranges
    return windows_by_ticker


def build_annual_coverage_report(
    news_df: pd.DataFrame,
    tickers: dict[str, str],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Build ticker-by-year headline counts for the requested research period."""
    scoped = news_df.loc[news_df["ticker"].isin(tickers)].copy()
    scoped["year"] = pd.to_datetime(
        scoped["publish_time"], errors="coerce", utc=True
    ).dt.year
    scoped = scoped.loc[scoped["year"].between(start_year, end_year)]
    counts = scoped.groupby(["ticker", "year"]).size().unstack(fill_value=0)
    counts = counts.reindex(
        index=list(tickers),
        columns=range(start_year, end_year + 1),
        fill_value=0,
    ).astype(int)
    report = counts.reset_index().rename(columns={"index": "ticker"})
    report.insert(1, "asset_name", report["ticker"].map(tickers))
    year_columns = list(range(start_year, end_year + 1))
    report["missing_years"] = report.apply(
        lambda row: ",".join(
            str(year) for year in year_columns if int(row[year]) == 0
        ),
        axis=1,
    )
    report["all_years_covered"] = report["missing_years"].eq("")
    return report


def print_annual_coverage_checker(
    annual_coverage: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> None:
    """Print an honest annual gap summary for the thesis period."""
    print("\n" + "=" * 96)
    print(f"ANNUAL HEADLINE COVERAGE: {start_year}-{end_year}")
    print("=" * 96)
    missing = annual_coverage.loc[
        ~annual_coverage["all_years_covered"],
        ["ticker", "missing_years"],
    ]
    if missing.empty:
        print(
            f"ALL YEARS COVERED: Every configured asset has at least one "
            f"headline in every year from {start_year} through {end_year}."
        )
        return
    total_pairs = len(annual_coverage) * (end_year - start_year + 1)
    missing_pairs = sum(
        len(value.split(","))
        for value in missing["missing_years"]
        if value
    )
    print(missing.to_string(index=False))
    print(
        f"\nMissing ticker-year pairs: {missing_pairs}/{total_pairs}. "
        "A zero means no stored headline for that asset-year."
    )


def build_coverage_report(
    news_df: pd.DataFrame,
    tickers: dict[str, str],
    start_ts: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    downloaded_news: pd.DataFrame | None = None,
    selected_tickers: set[str] | None = None,
) -> pd.DataFrame:
    """Build a readable per-asset coverage and current-run status report."""
    selected_tickers = selected_tickers or set()

    scoped_news = news_df.loc[
        news_df["ticker"].isin(tickers)
        & news_df["publish_time"].ge(start_ts)
        & news_df["publish_time"].lt(end_exclusive)
    ].copy()

    if downloaded_news is None or downloaded_news.empty:
        run_counts: dict[str, int] = {}
    else:
        current_run = downloaded_news.loc[
            downloaded_news["publish_time"].ge(start_ts)
            & downloaded_news["publish_time"].lt(end_exclusive)
        ]
        run_counts = current_run.groupby("ticker").size().astype(int).to_dict()

    rows: list[dict[str, Any]] = []
    for ticker, asset_name in tickers.items():
        asset_news = scoped_news.loc[scoped_news["ticker"].eq(ticker)]
        source_counts = asset_news["source"].value_counts().to_dict()
        total = int(len(asset_news))
        downloaded_this_run = int(run_counts.get(ticker, 0))

        yahoo_count = int(source_counts.get("yahoo_finance", 0))
        marketaux_count = int(source_counts.get("marketaux", 0))
        gdelt_count = int(source_counts.get("gdelt", 0))
        media_cloud_count = int(source_counts.get("media_cloud", 0))
        other_count = (
            total
            - yahoo_count
            - marketaux_count
            - gdelt_count
            - media_cloud_count
        )

        if downloaded_this_run:
            status = "DOWNLOADED_THIS_RUN"
        elif total:
            status = "ALREADY_AVAILABLE"
        else:
            status = "MISSING"

        providers_available = ",".join(sorted(source_counts))
        rows.append(
            {
                "ticker": ticker,
                "asset_name": asset_name,
                "status": status,
                "selected_this_run": ticker in selected_tickers,
                "downloaded_this_run": downloaded_this_run,
                "total_headlines": total,
                "yahoo_headlines": yahoo_count,
                "marketaux_headlines": marketaux_count,
                "gdelt_headlines": gdelt_count,
                "media_cloud_headlines": media_cloud_count,
                "other_headlines": other_count,
                "providers_available": providers_available,
                "first_publish_time": (
                    asset_news["publish_time"].min() if total else pd.NaT
                ),
                "last_publish_time": (
                    asset_news["publish_time"].max() if total else pd.NaT
                ),
            }
        )

    return pd.DataFrame(rows)


def print_coverage_checker(coverage: pd.DataFrame) -> None:
    """Print missing/detail rows, or one success line when all are covered."""
    print("\n" + "=" * 96)
    print("NEWS HEADLINE COVERAGE CHECKER")
    print("=" * 96)

    missing_mask = coverage["total_headlines"].eq(0)
    missing = coverage.loc[missing_mask, "ticker"].tolist()
    covered_count = int((~missing_mask).sum())

    if not missing:
        print(
            f"BASIC COVERAGE PASS: All {len(coverage)} configured assets have "
            "at least one valid headline somewhere in the configured date "
            "window. See the annual checker below for yearly gaps."
        )
        return

    display_columns = [
        "ticker",
        "asset_name",
        "status",
        "downloaded_this_run",
        "total_headlines",
        "yahoo_headlines",
        "marketaux_headlines",
        "gdelt_headlines",
        "media_cloud_headlines",
    ]
    print(coverage[display_columns].to_string(index=False))
    print(f"\nCovered assets: {covered_count}/{len(coverage)}")
    print(f"Missing assets: {missing}")


def write_audit_files(
    news_df: pd.DataFrame,
    tickers: dict[str, str],
    failures: list[dict[str, str]],
    start_ts: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    downloaded_news: pd.DataFrame | None = None,
    selected_tickers: set[str] | None = None,
) -> tuple[Path, Path, pd.DataFrame]:
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    coverage_path = data_dir / "news_coverage.csv"
    failures_path = data_dir / "news_collection_failures.csv"

    coverage = build_coverage_report(
        news_df,
        tickers,
        start_ts,
        end_exclusive,
        downloaded_news=downloaded_news,
        selected_tickers=selected_tickers,
    )
    atomic_write_csv(coverage, coverage_path)

    failures_df = pd.DataFrame(failures, columns=FAILURE_COLUMNS)
    atomic_write_csv(failures_df, failures_path)
    return coverage_path, failures_path, coverage


def select_tickers(requested: Iterable[str] | None) -> dict[str, str]:
    normalized = {str(ticker).strip().upper(): name for ticker, name in ALL_TICKERS.items()}
    if not requested:
        return normalized
    requested_list = [ticker.strip().upper() for ticker in requested]
    unknown = [ticker for ticker in requested_list if ticker not in normalized]
    if unknown:
        raise ValueError(f"Unknown ticker(s): {', '.join(unknown)}")
    return {ticker: normalized[ticker] for ticker in requested_list}


def parse_providers(raw_value: str) -> set[str]:
    providers = {item.strip().lower() for item in raw_value.split(",") if item.strip()}
    unknown = providers - SUPPORTED_PROVIDERS
    if unknown:
        raise ValueError(f"Unknown provider(s): {', '.join(sorted(unknown))}")
    if not providers:
        raise ValueError("At least one provider is required")
    return providers


def collect_news(
    tickers: dict[str, str],
    providers: set[str],
    start_ts: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    marketaux_pages: int,
    marketaux_budget_limit: int,
    gdelt_budget_limit: int,
    media_cloud_budget_limit: int,
    media_cloud_window_months: int,
    media_cloud_max_pages: int,
    media_cloud_pause_seconds: float,
    media_cloud_restart: bool,
    media_cloud_collection_ids: Iterable[int] | None,
    media_cloud_windows_by_ticker: dict[str, list[tuple[date, date]]] | None,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    marketaux_token = str(MARKETAUX_API_TOKEN or "").strip()
    marketaux_budget = RequestBudget(marketaux_budget_limit)
    gdelt_budget = RequestBudget(gdelt_budget_limit)
    gdelt_rate_limiter = RateLimiter(GDELT_SECONDS_BETWEEN_CALLS)
    failures: list[dict[str, str]] = []
    downloaded_rows: list[dict[str, Any]] = []
    media_cloud_state: MediaCloudState | None = None

    if "media_cloud" in providers:
        try:
            media_cloud_state = create_media_cloud_state(
                start_ts=start_ts,
                end_exclusive=end_exclusive,
                budget_limit=media_cloud_budget_limit,
                window_months=media_cloud_window_months,
                max_pages_per_window=media_cloud_max_pages,
                pause_seconds=media_cloud_pause_seconds,
                restart=media_cloud_restart,
                collection_ids=media_cloud_collection_ids,
            )
        except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
            failures.append(
                {"ticker": "*", "provider": MEDIA_CLOUD_SOURCE, "error": str(exc)}
            )
            print(f"Media Cloud skipped: {exc}", file=sys.stderr)

    if "marketaux" in providers and not marketaux_token:
        print(
            "Marketaux skipped: data_config.MARKETAUX_API_TOKEN is empty. "
            "Check MARKETAUX_API_TOKEN in .env. "
            "Yahoo and GDELT will still run.",
            file=sys.stderr,
        )

    if "gdelt" in providers and start_ts < (
        pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=GDELT_RECENT_WINDOW_DAYS)
    ):
        print(
            f"GDELT warning: only its recent ~{GDELT_RECENT_WINDOW_DAYS}-day window "
            "will be queried; older history requires a licensed/archive dataset.",
            file=sys.stderr,
        )

    total = len(tickers)
    marketaux_exhausted = False
    gdelt_exhausted = False
    media_cloud_exhausted = False

    for number, (ticker, configured_name) in enumerate(tickers.items(), start=1):
        query_name = str(configured_name).strip()
        print(f"\n[{number:02d}/{total}] {ticker} — {query_name}")

        if "yahoo" in providers:
            try:
                rows = fetch_yahoo_news(ticker, start_ts, end_exclusive)
                downloaded_rows.extend(rows)
                print(f"  Yahoo:    {len(rows):4d}")
            except RuntimeError as exc:
                failures.append({"ticker": ticker, "provider": "yahoo", "error": str(exc)})
                print(f"  Yahoo error: {exc}", file=sys.stderr)

        if "marketaux" in providers and marketaux_token and not marketaux_exhausted:
            try:
                rows = fetch_marketaux_news(
                    ticker,
                    query_name,
                    marketaux_token,
                    start_ts,
                    end_exclusive,
                    marketaux_budget,
                    marketaux_pages,
                )
                downloaded_rows.extend(rows)
                print(f"  Marketaux:{len(rows):4d}")
            except RuntimeError as exc:
                if "budget exhausted" in str(exc):
                    marketaux_exhausted = True
                failures.append(
                    {"ticker": ticker, "provider": "marketaux", "error": str(exc)}
                )
                print(f"  Marketaux error: {exc}", file=sys.stderr)

        if "gdelt" in providers and not gdelt_exhausted:
            try:
                rows = fetch_gdelt_news(
                    ticker,
                    query_name,
                    start_ts,
                    end_exclusive,
                    gdelt_budget,
                    gdelt_rate_limiter,
                )
                downloaded_rows.extend(rows)
                print(f"  GDELT:   {len(rows):4d}")
            except RuntimeError as exc:
                if "budget exhausted" in str(exc):
                    gdelt_exhausted = True
                failures.append({"ticker": ticker, "provider": "gdelt", "error": str(exc)})
                print(f"  GDELT error: {exc}", file=sys.stderr)

        if media_cloud_state is not None and not media_cloud_exhausted:
            ticker_windows = (
                media_cloud_windows_by_ticker.get(ticker, [])
                if media_cloud_windows_by_ticker is not None
                else media_cloud_state.windows
            )
            rows, media_failures, exhausted, processed_windows = (
                collect_media_cloud_ticker(
                    media_cloud_state,
                    ticker,
                    query_name,
                    windows=ticker_windows,
                )
            )
            downloaded_rows.extend(
                {column: row[column] for column in NEWS_COLUMNS}
                for row in rows
            )
            failures.extend(media_failures)
            media_cloud_exhausted = exhausted
            print(
                f"  MediaCloud:{len(rows):4d} "
                f"({processed_windows}/{len(ticker_windows)} archive "
                "windows processed)"
            )
            if exhausted:
                print(
                    "  Media Cloud request budget reached; rerun the same "
                    "command to resume.",
                    file=sys.stderr,
                )

        time.sleep(PAUSE_BETWEEN_TICKERS_SECONDS)

    usage = (
        f"Marketaux={marketaux_budget.used}/{marketaux_budget.limit}, "
        f"GDELT={gdelt_budget.used}/{gdelt_budget.limit}"
    )
    if media_cloud_state is not None:
        windows_by_ticker = media_cloud_windows_by_ticker or {
            ticker: media_cloud_state.windows for ticker in tickers
        }
        complete_windows = sum(
            media_cloud_progress_key(
                ticker,
                window_start,
                window_end,
                media_cloud_collection_ids_for_ticker(
                    media_cloud_state,
                    ticker,
                ),
            )
            in media_cloud_state.completed
            for ticker in tickers
            for window_start, window_end in windows_by_ticker[ticker]
        )
        total_windows = sum(len(windows) for windows in windows_by_ticker.values())
        usage += (
            f", MediaCloud={media_cloud_state.budget.used}/"
            f"{media_cloud_state.budget.limit}, "
            f"MediaCloud windows={complete_windows}/{total_windows}"
        )
    print(f"\nAPI usage: {usage}")
    return pd.DataFrame(downloaded_rows, columns=NEWS_COLUMNS), failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--providers",
        default="yahoo,marketaux,gdelt,media_cloud",
        help="comma-separated providers: yahoo,marketaux,gdelt,media_cloud",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        help="optional subset, for example SAP.DE ASML.AS NESN.SW",
    )
    parser.add_argument(
        "--marketaux-pages",
        type=int,
        default=DEFAULT_MARKETAUX_MAX_PAGES,
        help="pages per ticker; free responses currently contain up to 3 articles/page",
    )
    parser.add_argument(
        "--marketaux-budget",
        type=int,
        default=DEFAULT_MARKETAUX_DAILY_BUDGET,
        help="maximum Marketaux requests for this run",
    )
    parser.add_argument(
        "--gdelt-max-calls",
        type=int,
        default=DEFAULT_GDELT_MAX_CALLS,
        help="safety cap for GDELT requests, including interval splits",
    )
    parser.add_argument(
        "--media-cloud-budget",
        type=int,
        default=DEFAULT_MEDIA_CLOUD_REQUEST_BUDGET,
        help="maximum Media Cloud requests for this run",
    )
    parser.add_argument(
        "--media-cloud-window-months",
        type=int,
        default=DEFAULT_MEDIA_CLOUD_WINDOW_MONTHS,
        help="months per historical Media Cloud query window",
    )
    parser.add_argument(
        "--media-cloud-max-pages",
        type=int,
        default=DEFAULT_MEDIA_CLOUD_MAX_PAGES_PER_WINDOW,
        help="pagination safety cap for each Media Cloud date window",
    )
    parser.add_argument(
        "--media-cloud-pause-seconds",
        type=float,
        default=DEFAULT_MEDIA_CLOUD_PAUSE_SECONDS,
        help="pause between Media Cloud pagination calls",
    )
    parser.add_argument(
        "--media-cloud-restart",
        action="store_true",
        help="ignore Media Cloud progress; downloaded rows remain deduplicated",
    )
    parser.add_argument(
        "--media-cloud-collection-ids",
        nargs="+",
        type=int,
        default=None,
        metavar="ID",
        help=(
            "optional Media Cloud collection IDs; otherwise discover European "
            "national collections automatically"
        ),
    )
    parser.add_argument(
        "--missing-years-only",
        action="store_true",
        help=(
            "with Media Cloud, query only years that contain zero stored "
            "headlines, merging consecutive missing years into one range"
        ),
    )
    parser.add_argument(
        "--coverage-start-year",
        type=int,
        default=DEFAULT_COVERAGE_START_YEAR,
        help="first year used by annual coverage and --missing-years-only",
    )
    parser.add_argument(
        "--coverage-end-year",
        type=int,
        default=DEFAULT_COVERAGE_END_YEAR,
        help="last year used by annual coverage and --missing-years-only",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="show and save per-asset coverage without calling any news API",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        providers = parse_providers(args.providers)
        tickers = select_tickers(args.tickers)
        if args.marketaux_pages < 1:
            raise ValueError("--marketaux-pages must be at least 1")
        if (
            args.marketaux_budget < 1
            or args.gdelt_max_calls < 1
            or args.media_cloud_budget < 1
            or args.media_cloud_window_months < 1
            or args.media_cloud_max_pages < 1
        ):
            raise ValueError("API request budgets must be positive")
        if args.media_cloud_pause_seconds < 0:
            raise ValueError("--media-cloud-pause-seconds must not be negative")
        if args.coverage_start_year > args.coverage_end_year:
            raise ValueError(
                "--coverage-start-year must not be after --coverage-end-year"
            )
        start_ts, end_exclusive = get_date_window()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    allowed_end = end_exclusive - pd.Timedelta(microseconds=1)
    print(
        f"Collection window: {start_ts.date()} to {allowed_end.date()} (inclusive)\n"
        f"Providers: {', '.join(sorted(providers))}\n"
        f"Assets: {len(tickers)}"
    )

    existing_news = load_existing_news()
    if args.check_only:
        coverage_path, _, coverage = write_audit_files(
            existing_news,
            ALL_TICKERS,
            [],
            start_ts,
            end_exclusive,
        )
        print_coverage_checker(coverage)
        annual_coverage = build_annual_coverage_report(
            existing_news,
            ALL_TICKERS,
            args.coverage_start_year,
            args.coverage_end_year,
        )
        annual_path = Path(DATA_DIR) / "news_annual_coverage.csv"
        atomic_write_csv(annual_coverage, annual_path)
        print_annual_coverage_checker(
            annual_coverage,
            args.coverage_start_year,
            args.coverage_end_year,
        )
        print(f"\nCoverage report: {coverage_path}")
        print(f"Annual coverage: {annual_path}")
        print("No API requests were made.")
        return 0

    if "media_cloud" in providers:
        recovered_media_cloud = recover_media_cloud_rows()
        if not recovered_media_cloud.empty:
            existing_news = merge_news(existing_news, recovered_media_cloud)
            print(
                f"Recovered {len(recovered_media_cloud):,} checkpointed "
                "Media Cloud rows before collection"
            )

    media_cloud_windows_by_ticker = None
    if args.missing_years_only:
        if "media_cloud" not in providers:
            print(
                "--missing-years-only affects only the Media Cloud provider.",
                file=sys.stderr,
            )
        media_cloud_windows_by_ticker = build_missing_year_windows(
            existing_news,
            tickers,
            args.coverage_start_year,
            args.coverage_end_year,
        )
        missing_range_count = sum(
            len(windows) for windows in media_cloud_windows_by_ticker.values()
        )
        missing_year_count = sum(
            window_end.year - window_start.year + 1
            for windows in media_cloud_windows_by_ticker.values()
            for window_start, window_end in windows
        )
        print(
            f"Missing-year mode: {missing_year_count} missing ticker-years "
            f"merged into {missing_range_count} minimal date ranges for "
            f"{args.coverage_start_year}-{args.coverage_end_year}."
        )

    downloaded_news, failures = collect_news(
        tickers=tickers,
        providers=providers,
        start_ts=start_ts,
        end_exclusive=end_exclusive,
        marketaux_pages=args.marketaux_pages,
        marketaux_budget_limit=args.marketaux_budget,
        gdelt_budget_limit=args.gdelt_max_calls,
        media_cloud_budget_limit=args.media_cloud_budget,
        media_cloud_window_months=args.media_cloud_window_months,
        media_cloud_max_pages=args.media_cloud_max_pages,
        media_cloud_pause_seconds=args.media_cloud_pause_seconds,
        media_cloud_restart=args.media_cloud_restart,
        media_cloud_collection_ids=args.media_cloud_collection_ids,
        media_cloud_windows_by_ticker=media_cloud_windows_by_ticker,
    )
    merged_news = merge_news(existing_news, downloaded_news)
    atomic_write_csv(merged_news, Path(NEWS_FILE))
    coverage_path, failures_path, coverage = write_audit_files(
        merged_news,
        ALL_TICKERS,
        failures,
        start_ts,
        end_exclusive,
        downloaded_news=downloaded_news,
        selected_tickers=set(tickers),
    )

    print(f"\nDownloaded rows before deduplication: {len(downloaded_news):,}")
    print(f"Saved total unique rows:             {len(merged_news):,}")
    print(f"News file:       {NEWS_FILE}")
    print(f"Coverage report: {coverage_path}")
    print(f"Failure report:  {failures_path}")
    print_coverage_checker(coverage)
    annual_coverage = build_annual_coverage_report(
        merged_news,
        ALL_TICKERS,
        args.coverage_start_year,
        args.coverage_end_year,
    )
    annual_path = Path(DATA_DIR) / "news_annual_coverage.csv"
    atomic_write_csv(annual_coverage, annual_path)
    print_annual_coverage_checker(
        annual_coverage,
        args.coverage_start_year,
        args.coverage_end_year,
    )
    print(f"Annual coverage: {annual_path}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
