"""
complete_asset_full_comparison.py

Produces a full 2019–2025 forecast-comparison table for every configured
asset with sufficiently complete out-of-sample coverage.

Output columns:

    Date
    Ticker
    Actual
    XGBoost
    XGBoost+Sentiment
    Markowitz (implied)

Asset selection is based only on data availability—not forecast accuracy.
This avoids selecting assets because they performed well out of sample.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from data_config import (
    ALL_TICKERS,
    CLEAN_PRICES_FILE,
)
from backtest_config import (
    BACKTEST_START,
    BACKTEST_END,
    FEATURE_SETS,
    RESULTS_DIR,
)
from forecast_model import (
    load_panel,
    ReturnForecaster,
)
from Backtest import get_rebalance_dates


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Use 1.00 to retain only assets with observations at every rebalancing date.
#
# If this is too strict, use 0.95, but document that decision in the thesis.
MIN_COVERAGE_RATIO = 0.95

# None means retain every qualifying asset.
# You may set this to an integer such as 20 if required.
MAX_ASSETS = None

# If strict MIN_COVERAGE_RATIO filtering leaves fewer than this many
# assets, the run no longer fails outright. Selection is widened in
# two steps to keep as many assets as possible:
#   1. Every ticker with coverage_ratio >= MIN_COVERAGE_RATIO_FALLBACK
#      is added (a looser bar than the strict MIN_COVERAGE_RATIO).
#   2. FALLBACK_TICKERS are force-included regardless of their
#      coverage ratio, as long as they have at least some usable data.
# Document this relaxation in the thesis methodology/limitations if
# it triggers, since not every selected asset will have full coverage.
MIN_ASSETS = None

# Looser coverage bar used only when the strict MIN_COVERAGE_RATIO
# filter leaves fewer than MIN_ASSETS assets. Lower this (e.g. to
# 0.80) to pull in even more assets; raise it to be stricter.
MIN_COVERAGE_RATIO_FALLBACK = 0.90

# Tickers that must be included whenever the fallback path triggers,
# regardless of their individual coverage ratio.
FALLBACK_TICKERS = [
    "SAP.DE",
    "SIE.DE",
    "MC.PA",
    "NESN.SW",
    "EXH8.DE",
]

MARKOWITZ_LOOKBACK_DAYS = 756
MIN_MARKOWITZ_OBSERVATIONS = 252
FORECAST_HORIZON_DAYS = 21

RESULTS_PATH = Path(RESULTS_DIR)

OUTPUT_FILE = (
    RESULTS_PATH
    / "complete_asset_full_comparison.csv"
)

COVERAGE_FILE = (
    RESULTS_PATH
    / "complete_asset_comparison_coverage.csv"
)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

started = time.time()

RESULTS_PATH.mkdir(
    parents=True,
    exist_ok=True,
)

print("=" * 80)
print("COMPLETE-ASSET FORECAST COMPARISON")
print("=" * 80)

panel = load_panel().copy()

panel["date"] = pd.to_datetime(
    panel["date"],
    errors="coerce",
)

panel = panel.dropna(
    subset=["date", "ticker"],
)

trading_dates = pd.DatetimeIndex(
    sorted(panel["date"].unique())
)

prices = pd.read_csv(
    CLEAN_PRICES_FILE,
    index_col=0,
    parse_dates=True,
)

prices.index = pd.to_datetime(
    prices.index,
    errors="coerce",
)

prices = prices.sort_index()


# ---------------------------------------------------------------------------
# Determine rebalancing dates
# ---------------------------------------------------------------------------

rebalance_dates = get_rebalance_dates(
    prices.index,
    BACKTEST_START,
    BACKTEST_END,
)

rebalance_dates = pd.DatetimeIndex(
    pd.to_datetime(rebalance_dates)
)

if len(rebalance_dates) == 0:
    raise RuntimeError(
        "No rebalancing dates were generated. "
        "Check BACKTEST_START, BACKTEST_END and the price index."
    )

print(
    f"\nEvaluation period: "
    f"{rebalance_dates[0].date()} to "
    f"{rebalance_dates[-1].date()}"
)

print(
    f"Expected rebalancing dates: "
    f"{len(rebalance_dates)}"
)


# ---------------------------------------------------------------------------
# Identify candidate assets
# ---------------------------------------------------------------------------

configured_tickers = list(ALL_TICKERS)

panel_tickers = set(
    panel["ticker"].dropna().astype(str)
)

price_tickers = set(prices.columns)

candidate_tickers = [
    ticker
    for ticker in configured_tickers
    if (
        ticker in panel_tickers
        and ticker in price_tickers
    )
]

missing_from_panel = [
    ticker
    for ticker in configured_tickers
    if ticker not in panel_tickers
]

missing_from_prices = [
    ticker
    for ticker in configured_tickers
    if ticker not in price_tickers
]

print(
    f"Configured assets: "
    f"{len(configured_tickers)}"
)

print(
    f"Candidate assets with both panel and prices: "
    f"{len(candidate_tickers)}"
)

if missing_from_panel:
    print(
        "\nMissing from model panel:",
        missing_from_panel,
    )

if missing_from_prices:
    print(
        "\nMissing from price data:",
        missing_from_prices,
    )

if not candidate_tickers:
    raise RuntimeError(
        "No configured assets exist in both the "
        "model panel and price file."
    )


# ---------------------------------------------------------------------------
# Prepare actual-return lookup
# ---------------------------------------------------------------------------

actual_lookup = (
    panel[
        [
            "date",
            "ticker",
            "target",
        ]
    ]
    .drop_duplicates(
        subset=["date", "ticker"],
        keep="last",
    )
    .set_index(
        ["date", "ticker"]
    )["target"]
)


# ---------------------------------------------------------------------------
# Initialise forecasting models
# ---------------------------------------------------------------------------

required_models = (
    "xgb",
    "xgb_sentiment",
)

missing_feature_sets = [
    model
    for model in required_models
    if model not in FEATURE_SETS
]

if missing_feature_sets:
    raise KeyError(
        "Missing required FEATURE_SETS entries: "
        f"{missing_feature_sets}"
    )

forecasters = {
    model: ReturnForecaster(
        FEATURE_SETS[model]
    )
    for model in required_models
}


# ---------------------------------------------------------------------------
# Generate out-of-sample forecasts
# ---------------------------------------------------------------------------

rows = []

print(
    "\nGenerating forecasts using the full "
    "cross-sectional training panel..."
)

for step, as_of in enumerate(
    rebalance_dates,
    start=1,
):
    as_of = pd.Timestamp(as_of)

    predictions = {}

    for model in required_models:
        predictions[model] = (
            forecasters[model].fit_predict(
                panel,
                trading_dates,
                as_of,
                refit=True,
            )
        )

    for ticker in candidate_tickers:
        actual = actual_lookup.get(
            (as_of, ticker),
            np.nan,
        )

        if pd.isna(actual):
            continue

        xgb_predictions = predictions["xgb"]
        sentiment_predictions = predictions[
            "xgb_sentiment"
        ]

        if (
            xgb_predictions is None
            or sentiment_predictions is None
        ):
            continue

        if (
            ticker not in xgb_predictions.index
            or ticker not in sentiment_predictions.index
        ):
            continue

        xgb_prediction = xgb_predictions.loc[
            ticker
        ]

        sentiment_prediction = (
            sentiment_predictions.loc[ticker]
        )

        if (
            pd.isna(xgb_prediction)
            or pd.isna(sentiment_prediction)
        ):
            continue

        rows.append(
            {
                "date": as_of,
                "ticker": ticker,
                "actual": float(actual),
                "xgb_predicted": float(
                    xgb_prediction
                ),
                "xgb_sentiment_predicted": float(
                    sentiment_prediction
                ),
            }
        )

    if (
        step == 1
        or step % 12 == 0
        or step == len(rebalance_dates)
    ):
        print(
            f"  {as_of.date()} "
            f"({step}/{len(rebalance_dates)})"
        )


forecast_data = pd.DataFrame(rows)

if forecast_data.empty:
    raise RuntimeError(
        "No forecasts were generated. Check the panel, "
        "feature sets and backtest dates."
    )


# ---------------------------------------------------------------------------
# Calculate Markowitz-implied forecasts
# ---------------------------------------------------------------------------

available_price_tickers = [
    ticker
    for ticker in candidate_tickers
    if ticker in prices.columns
]

log_prices = np.log(
    prices[available_price_tickers]
)

daily_log_returns = (
    log_prices
    .diff()
    .replace(
        [np.inf, -np.inf],
        np.nan,
    )
)

markowitz_rows = []

print(
    "\nCalculating Markowitz-implied returns..."
)

unique_forecast_dates = sorted(
    forecast_data["date"].unique()
)

for as_of in unique_forecast_dates:
    as_of = pd.Timestamp(as_of)

    window = (
        daily_log_returns
        .loc[:as_of]
        .tail(MARKOWITZ_LOOKBACK_DAYS)
    )

    observation_counts = window.count()

    implied_returns = (
        window.mean()
        * FORECAST_HORIZON_DAYS
    )

    for ticker in available_price_tickers:
        if (
            observation_counts.get(ticker, 0)
            < MIN_MARKOWITZ_OBSERVATIONS
        ):
            continue

        implied_value = implied_returns.get(
            ticker,
            np.nan,
        )

        if pd.isna(implied_value):
            continue

        markowitz_rows.append(
            {
                "date": as_of,
                "ticker": ticker,
                "markowitz_implied": float(
                    implied_value
                ),
            }
        )


markowitz_data = pd.DataFrame(
    markowitz_rows
)

if markowitz_data.empty:
    raise RuntimeError(
        "No Markowitz-implied forecasts were generated."
    )


# ---------------------------------------------------------------------------
# Combine forecasts
# ---------------------------------------------------------------------------

comparison = forecast_data.merge(
    markowitz_data,
    on=["date", "ticker"],
    how="left",
)

required_columns = [
    "actual",
    "xgb_predicted",
    "xgb_sentiment_predicted",
    "markowitz_implied",
]

comparison = comparison.dropna(
    subset=required_columns
)


# ---------------------------------------------------------------------------
# Measure asset coverage
# ---------------------------------------------------------------------------

expected_dates = len(rebalance_dates)

coverage = (
    comparison.groupby("ticker")
    .agg(
        usable_rebalancing_dates=(
            "date",
            "nunique",
        ),
        first_forecast=(
            "date",
            "min",
        ),
        last_forecast=(
            "date",
            "max",
        ),
    )
)

coverage["expected_rebalancing_dates"] = (
    expected_dates
)

coverage["coverage_ratio"] = (
    coverage["usable_rebalancing_dates"]
    / expected_dates
)

coverage["complete"] = (
    coverage["coverage_ratio"]
    >= MIN_COVERAGE_RATIO
)

coverage = coverage.sort_values(
    [
        "coverage_ratio",
        "usable_rebalancing_dates",
    ],
    ascending=False,
)


# ---------------------------------------------------------------------------
# Select assets based on coverage, with a minimum-asset fallback
# ---------------------------------------------------------------------------

selected_tickers = (
    coverage[
        coverage["complete"]
    ]
    .index
    .tolist()
)

if MAX_ASSETS is not None:
    selected_tickers = selected_tickers[
        :MAX_ASSETS
    ]

# If the strict MIN_COVERAGE_RATIO filter leaves fewer than MIN_ASSETS
# tickers, widen the selection in two steps instead of failing the
# whole run:
#   1. Pull in every ticker at or above MIN_COVERAGE_RATIO_FALLBACK
#      (a looser bar than the strict MIN_COVERAGE_RATIO), to keep as
#      many assets as possible.
#   2. Force-include FALLBACK_TICKERS regardless of their individual
#      coverage ratio, as long as they have at least some data.
# This relaxation is logged to stdout and should be noted in the
# thesis methodology/limitations if it triggers, since it means not
# every selected asset has full coverage.
if len(selected_tickers) < MIN_ASSETS:
    print(
        f"\nOnly {len(selected_tickers)} asset(s) met strict "
        f"MIN_COVERAGE_RATIO={MIN_COVERAGE_RATIO:.0%}. "
        "Widening selection to include as many assets as possible "
        f"(coverage_ratio >= {MIN_COVERAGE_RATIO_FALLBACK:.0%}), "
        f"plus the designated FALLBACK_TICKERS."
    )

    relaxed_tickers = (
        coverage[
            coverage["coverage_ratio"]
            >= MIN_COVERAGE_RATIO_FALLBACK
        ]
        .index
        .tolist()
    )

    selected_tickers = sorted(
        set(selected_tickers) | set(relaxed_tickers)
    )

forced_tickers = [
    ticker
    for ticker in FALLBACK_TICKERS
    if ticker in coverage.index
]

missing_fallback = [
    ticker
    for ticker in FALLBACK_TICKERS
    if ticker not in coverage.index
]

if missing_fallback:
    print(
        "  fallback tickers with no usable coverage at all "
        f"(skipped): {missing_fallback}"
    )

if forced_tickers:
    selected_tickers = sorted(
        set(selected_tickers) | set(forced_tickers)
    )

# If even the relaxed threshold plus the forced tickers still don't
# reach MIN_ASSETS (e.g. MIN_COVERAGE_RATIO_FALLBACK was set too high
# for this dataset), backfill with whichever remaining tickers have
# the next-highest coverage ratio, however low.
if len(selected_tickers) < MIN_ASSETS:
    remaining_slots = MIN_ASSETS - len(selected_tickers)

    backfill = [
        ticker
        for ticker in coverage.sort_values(
            "coverage_ratio",
            ascending=False,
        ).index
        if ticker not in selected_tickers
    ][:remaining_slots]

    selected_tickers = sorted(
        set(selected_tickers) | set(backfill)
    )

if selected_tickers and len(selected_tickers) > len(
    coverage[coverage["complete"]]
):
    print(
        f"\nFinal selection: {len(selected_tickers)} asset(s) "
        "(includes tickers below strict MIN_COVERAGE_RATIO):"
    )

    for ticker in selected_tickers:
        ticker_ratio = coverage.loc[
            ticker,
            "coverage_ratio",
        ]

        below_threshold = (
            ticker_ratio < MIN_COVERAGE_RATIO
        )

        print(
            f"  {ticker:<12} "
            f"{ticker_ratio:.1%} coverage"
            + (
                "  (below MIN_COVERAGE_RATIO, kept for fallback)"
                if below_threshold
                else ""
            )
        )

if not selected_tickers:
    best_coverage = (
        coverage["coverage_ratio"].max()
        if not coverage.empty
        else 0
    )

    raise RuntimeError(
        "No asset satisfied MIN_COVERAGE_RATIO="
        f"{MIN_COVERAGE_RATIO:.0%}, and no fallback tickers "
        "had any usable coverage either. "
        f"The best available coverage was "
        f"{best_coverage:.1%}. "
        "Inspect the coverage report, lower "
        "MIN_COVERAGE_RATIO_FALLBACK, or update FALLBACK_TICKERS."
    )

coverage["selected"] = (
    coverage.index.isin(selected_tickers)
)

coverage.to_csv(
    COVERAGE_FILE,
    index=True,
)


# ---------------------------------------------------------------------------
# Create final output
# ---------------------------------------------------------------------------

final = comparison[
    comparison["ticker"].isin(
        selected_tickers
    )
].copy()

final = final[
    [
        "date",
        "ticker",
        "actual",
        "xgb_predicted",
        "xgb_sentiment_predicted",
        "markowitz_implied",
    ]
]

final.columns = [
    "Date",
    "Ticker",
    "Actual",
    "XGBoost",
    "XGBoost+Sentiment",
    "Markowitz (implied)",
]

final = final.sort_values(
    ["Ticker", "Date"]
)

final.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ---------------------------------------------------------------------------
# Print completion summary
# ---------------------------------------------------------------------------

runtime = time.time() - started

print("\n" + "=" * 80)
print("COLLECTION COMPLETE")
print("=" * 80)

print(
    f"Selected assets: "
    f"{len(selected_tickers)}"
)

print(
    f"Saved comparison rows: "
    f"{len(final):,}"
)

print(
    f"Coverage requirement: "
    f"{MIN_COVERAGE_RATIO:.0%}"
)

print(
    f"Comparison file: "
    f"{OUTPUT_FILE}"
)

print(
    f"Coverage report: "
    f"{COVERAGE_FILE}"
)

print(
    f"Runtime: "
    f"{runtime:.0f} seconds"
)

print(
    "\nSelected tickers:"
)

for ticker in selected_tickers:
    ticker_coverage = coverage.loc[
        ticker,
        "coverage_ratio",
    ]

    print(
        f"  {ticker:<12} "
        f"{ticker_coverage:.1%} coverage"
    )


# ---------------------------------------------------------------------------
# Mean absolute error summary
# ---------------------------------------------------------------------------

print(
    "\n--- Mean absolute error by asset ---"
)

mae_rows = []

for ticker in selected_tickers:
    subset = final[
        final["Ticker"] == ticker
    ]

    if subset.empty:
        continue

    xgb_mae = (
        subset["XGBoost"]
        - subset["Actual"]
    ).abs().mean()

    sentiment_mae = (
        subset["XGBoost+Sentiment"]
        - subset["Actual"]
    ).abs().mean()

    markowitz_mae = (
        subset["Markowitz (implied)"]
        - subset["Actual"]
    ).abs().mean()

    sentiment_improvement = (
        xgb_mae
        - sentiment_mae
    )

    mae_rows.append(
        {
            "Ticker": ticker,
            "Observations": len(subset),
            "XGBoost MAE": xgb_mae,
            "XGBoost+Sentiment MAE": sentiment_mae,
            "Markowitz MAE": markowitz_mae,
            "Sentiment MAE improvement": (
                sentiment_improvement
            ),
        }
    )

    print(
        f"  {ticker:<12} "
        f"XGBoost={xgb_mae:.4f}  "
        f"XGBoost+Sentiment={sentiment_mae:.4f}  "
        f"Markowitz={markowitz_mae:.4f}  "
        f"Improvement={sentiment_improvement:+.4f}"
    )


mae_summary = pd.DataFrame(mae_rows)

mae_summary_file = (
    RESULTS_PATH
    / "complete_asset_mae_summary.csv"
)

mae_summary.to_csv(
    mae_summary_file,
    index=False,
)

print(
    f"\nMAE summary: "
    f"{mae_summary_file}"
)