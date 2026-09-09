# ============================================================
# JANUARY 2026 FORECAST — FIVE SELECTED ASSETS
# ============================================================

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from data_config import CLEAN_PRICES_FILE
from backtest_config import (
    FEATURE_SETS,
    FORECAST_HORIZON,
    RESULTS_DIR,
)
from forecast_model import load_panel, ReturnForecaster


# ------------------------------------------------------------
# 1. Selected assets
# ------------------------------------------------------------

ASSETS = {
    "EXH8.DE": (
        "iShares STOXX Europe 600 Retail UCITS ETF",
        "ETF",
    ),

    "SAP.DE": (
        "SAP SE",
        "Stock",
    ),

    "SIE.DE": (
        "Siemens AG",
        "Stock",
    ),

    "MC.PA": (
        "LVMH",
        "Stock",
    ),

    "NESN.SW": (
        "Nestlé SA",
        "Stock",
    ),
}

SELECTED = list(ASSETS)

# yfinance end date is exclusive. The February buffer ensures
# every exchange can supply 21 valid sessions.
ACTUAL_START = "2026-01-01"
ACTUAL_END = "2026-02-10"
DOWNLOAD_RETRIES = 3


# ------------------------------------------------------------
# 2. Load model data
# ------------------------------------------------------------

panel = load_panel()

prices = pd.read_csv(
    CLEAN_PRICES_FILE,
    index_col=0,
    parse_dates=True,
).sort_index()

trading_dates = pd.DatetimeIndex(
    sorted(panel["date"].unique())
)


# ------------------------------------------------------------
# 3. Validate selected assets
# ------------------------------------------------------------

panel_tickers = set(panel["ticker"].unique())
price_tickers = set(prices.columns)

missing_tickers = [
    ticker
    for ticker in SELECTED
    if ticker not in panel_tickers
    or ticker not in price_tickers
]

if missing_tickers:
    raise ValueError(
        "The following selected assets are missing from the "
        f"price or feature data: {missing_tickers}"
    )

print(f"Selected assets ({len(SELECTED)}):")
print(", ".join(SELECTED))


# ------------------------------------------------------------
# 4. Select the final available date in 2025
# ------------------------------------------------------------

available_2025_dates = trading_dates[
    trading_dates <= pd.Timestamp("2025-12-31")
]

if len(available_2025_dates) == 0:
    raise ValueError(
        "No feature observations were found before "
        "31 December 2025."
    )

forecast_date = available_2025_dates[-1]

print()
print("Forecast produced using information through:")
print(forecast_date.date())
print(
    f"Forecast horizon: "
    f"{FORECAST_HORIZON} trading sessions"
)


# ------------------------------------------------------------
# 5. Train XGBoost models
# ------------------------------------------------------------

forecasters = {
    model_name: ReturnForecaster(feature_columns)
    for model_name, feature_columns
    in FEATURE_SETS.items()
}

model_predictions = {}

for model_name, forecaster in forecasters.items():

    predictions = forecaster.fit_predict(
        panel=panel,
        trading_dates=trading_dates,
        rebalance_date=forecast_date,
        universe=SELECTED,
        refit=True,
    )

    if predictions is None:
        raise RuntimeError(
            f"{model_name} could not produce predictions."
        )

    model_predictions[model_name] = predictions

    print(
        f"{model_name:<15} | "
        f"training rows: {forecaster.last_train_rows:,} | "
        f"last admitted training date: "
        f"{forecaster.last_train_end.date()}"
    )


# ------------------------------------------------------------
# 6. Calculate Markowitz forecast
# ------------------------------------------------------------

daily_log_returns = np.log(
    prices[SELECTED]
).diff()

markowitz_window = daily_log_returns.loc[
    :forecast_date
].tail(756)

if len(markowitz_window) < 252:
    raise ValueError(
        "Not enough historical observations for the "
        "Markowitz forecast."
    )

markowitz_log_forecast = (
    markowitz_window.mean()
    * FORECAST_HORIZON
)


# ------------------------------------------------------------
# 7. Yahoo Finance download helpers
# ------------------------------------------------------------

def extract_close_series(download_frame, ticker):
    """
    Extract the Close-price series from either possible
    yfinance column format.
    """

    if download_frame is None or download_frame.empty:
        return None

    if isinstance(download_frame.columns, pd.MultiIndex):

        if (
            "Close"
            not in download_frame.columns.get_level_values(0)
        ):
            return None

        close = download_frame["Close"]

        if isinstance(close, pd.Series):
            return close.rename(ticker)

        if ticker in close.columns:
            return close[ticker].rename(ticker)

        if close.shape[1] == 1:
            return close.iloc[:, 0].rename(ticker)

        return None

    if "Close" not in download_frame.columns:
        return None

    return download_frame["Close"].rename(ticker)


def normalize_close_series(close, ticker):
    """
    Normalize dates, timezones, duplicates and missing values.
    """

    close = close.copy().rename(ticker).dropna()

    close.index = pd.DatetimeIndex(
        pd.to_datetime(close.index)
    )

    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)

    close = close[
        ~close.index.duplicated(keep="last")
    ]

    return close.sort_index()


def download_one_ticker(ticker):
    """
    Download one ticker independently with retry logic.
    """

    last_error = None

    for attempt in range(
        1,
        DOWNLOAD_RETRIES + 1,
    ):

        try:
            frame = yf.download(
                tickers=ticker,
                start=ACTUAL_START,
                end=ACTUAL_END,
                auto_adjust=True,
                progress=False,
                group_by="column",
                threads=False,
            )

            close = extract_close_series(
                frame,
                ticker,
            )

            if (
                close is not None
                and not close.dropna().empty
            ):
                return normalize_close_series(
                    close,
                    ticker,
                )

            last_error = (
                "empty response or missing Close column"
            )

        except Exception as exc:
            last_error = str(exc)

        print(
            f"  {ticker}: attempt "
            f"{attempt}/{DOWNLOAD_RETRIES} failed "
            f"({last_error})."
        )

        if attempt < DOWNLOAD_RETRIES:
            time.sleep(2 * attempt)

    # Fallback request
    try:
        history = yf.Ticker(ticker).history(
            start=ACTUAL_START,
            end=ACTUAL_END,
            auto_adjust=True,
        )

        if (
            not history.empty
            and "Close" in history.columns
        ):
            return normalize_close_series(
                history["Close"],
                ticker,
            )

    except Exception as exc:
        last_error = str(exc)

    print(
        f"WARNING: all Yahoo Finance requests failed "
        f"for {ticker}: {last_error}"
    )

    return pd.Series(
        dtype=float,
        name=ticker,
    )


# ------------------------------------------------------------
# 8. Download January/February 2026 prices
# ------------------------------------------------------------

print()
print(
    f"Downloading adjusted prices from "
    f"{ACTUAL_START} to {ACTUAL_END} ..."
)

downloaded_series = {}

for ticker in SELECTED:

    close = download_one_ticker(ticker)
    downloaded_series[ticker] = close

    print(
        f"  {ticker}: "
        f"{close.dropna().shape[0]} "
        "valid observations downloaded"
    )

actual_prices = pd.concat(
    downloaded_series.values(),
    axis=1,
).sort_index()

if actual_prices.dropna(how="all").empty:
    raise RuntimeError(
        "Yahoo Finance returned no January/February "
        "2026 prices."
    )

os.makedirs(
    RESULTS_DIR,
    exist_ok=True,
)

actual_price_file = os.path.join(
    RESULTS_DIR,
    "january_2026_actual_prices.csv",
)

actual_prices.to_csv(actual_price_file)

print(
    f"Saved downloaded prices to "
    f"{actual_price_file}"
)


# ------------------------------------------------------------
# 9. Calculate actual returns and prices
# ------------------------------------------------------------

information_prices = {}
actual_returns = {}
actual_ending_prices = {}
actual_end_dates = {}
actual_session_counts = {}

for ticker in SELECTED:

    historical_prices = (
        prices.loc[:forecast_date, ticker]
        .dropna()
    )

    if historical_prices.empty:
        information_prices[ticker] = np.nan
    else:
        information_prices[ticker] = float(
            historical_prices.iloc[-1]
        )

    if ticker not in actual_prices.columns:

        actual_returns[ticker] = np.nan
        actual_ending_prices[ticker] = np.nan
        actual_end_dates[ticker] = pd.NaT
        actual_session_counts[ticker] = 0
        continue

    realized = (
        actual_prices[ticker]
        .dropna()
        .iloc[:FORECAST_HORIZON]
    )

    actual_session_counts[ticker] = len(realized)

    if len(realized) < FORECAST_HORIZON:

        print(
            f"WARNING: {ticker} has only "
            f"{len(realized)} valid sessions; "
            f"{FORECAST_HORIZON} are required."
        )

        actual_returns[ticker] = np.nan
        actual_ending_prices[ticker] = np.nan

        actual_end_dates[ticker] = (
            realized.index[-1]
            if len(realized)
            else pd.NaT
        )

        continue

    p0 = information_prices[ticker]
    p1 = float(
        realized.iloc[
            FORECAST_HORIZON - 1
        ]
    )

    actual_ending_prices[ticker] = p1
    actual_returns[ticker] = p1 / p0 - 1.0

    actual_end_dates[ticker] = (
        realized.index[
            FORECAST_HORIZON - 1
        ]
    )


# ------------------------------------------------------------
# 10. Build forecast and price table
# ------------------------------------------------------------

forecast_rows = []

for ticker in SELECTED:

    asset_name, asset_type = ASSETS[ticker]

    if ticker not in model_predictions["xgb"].index:
        print(
            f"Skipping {ticker}: "
            "XGBoost prediction unavailable."
        )
        continue

    if (
        ticker
        not in model_predictions[
            "xgb_sentiment"
        ].index
    ):
        print(
            f"Skipping {ticker}: sentiment-model "
            "prediction unavailable."
        )
        continue

    p0 = information_prices[ticker]

    xgb_log_return = float(
        model_predictions["xgb"][ticker]
    )

    sentiment_log_return = float(
        model_predictions[
            "xgb_sentiment"
        ][ticker]
    )

    markowitz_log_return = float(
        markowitz_log_forecast[ticker]
    )

    xgb_return = np.expm1(
        xgb_log_return
    )

    sentiment_return = np.expm1(
        sentiment_log_return
    )

    markowitz_return = np.expm1(
        markowitz_log_return
    )

    # Forecast price = information price × exp(log-return forecast)
    xgb_predicted_price = (
        p0 * np.exp(xgb_log_return)
    )

    sentiment_predicted_price = (
        p0 * np.exp(sentiment_log_return)
    )

    markowitz_predicted_price = (
        p0 * np.exp(markowitz_log_return)
    )

    forecast_rows.append(
        {
            "Type": asset_type,
            "Asset": asset_name,
            "Ticker": ticker,
            "Information date": forecast_date,
            "Actual end date": (
                actual_end_dates.get(
                    ticker,
                    pd.NaT,
                )
            ),
            "Actual sessions": (
                actual_session_counts.get(
                    ticker,
                    0,
                )
            ),

            # Prices
            "Information price": p0,
            "XGBoost predicted price": (
                xgb_predicted_price
            ),
            "Sentiment predicted price": (
                sentiment_predicted_price
            ),
            "Markowitz predicted price": (
                markowitz_predicted_price
            ),
            "Actual price": (
                actual_ending_prices.get(
                    ticker,
                    np.nan,
                )
            ),

            # Returns
            "XGBoost return": xgb_return,
            "Sentiment return": sentiment_return,
            "Markowitz return": markowitz_return,
            "Actual return": (
                actual_returns.get(
                    ticker,
                    np.nan,
                )
            ),
            "Sentiment adjustment": (
                sentiment_return
                - xgb_return
            ),
        }
    )

january_2026_forecast = pd.DataFrame(
    forecast_rows
)

if january_2026_forecast.empty:
    raise RuntimeError(
        "No January 2026 forecast rows were produced."
    )

january_2026_forecast[
    "Information date"
] = pd.to_datetime(
    january_2026_forecast[
        "Information date"
    ]
).dt.strftime("%d %b %Y")

january_2026_forecast[
    "Actual end date"
] = pd.to_datetime(
    january_2026_forecast[
        "Actual end date"
    ]
).dt.strftime("%d %b %Y")


# ------------------------------------------------------------
# 11. Save CSV
# ------------------------------------------------------------

forecast_file = os.path.join(
    RESULTS_DIR,
    "january_2026_forecast_prices_vs_actual.csv",
)

january_2026_forecast.to_csv(
    forecast_file,
    index=False,
)

print()
print(
    f"Saved forecast comparison to "
    f"{forecast_file}"
)


# ------------------------------------------------------------
# 12. Print a readable Terminal table
# ------------------------------------------------------------

terminal_columns = [
    "Ticker",
    "Information price",
    "XGBoost predicted price",
    "Sentiment predicted price",
    "Markowitz predicted price",
    "Actual price",
    "XGBoost return",
    "Sentiment return",
    "Markowitz return",
    "Actual return",
]

terminal_table = (
    january_2026_forecast[
        terminal_columns
    ].copy()
)

price_columns = [
    "Information price",
    "XGBoost predicted price",
    "Sentiment predicted price",
    "Markowitz predicted price",
    "Actual price",
]

return_columns = [
    "XGBoost return",
    "Sentiment return",
    "Markowitz return",
    "Actual return",
]

for column in price_columns:
    terminal_table[column] = (
        terminal_table[column].map(
            lambda value: (
                "—"
                if pd.isna(value)
                else f"€{value:,.2f}"
            )
        )
    )

for column in return_columns:
    terminal_table[column] = (
        terminal_table[column].map(
            lambda value: (
                "—"
                if pd.isna(value)
                else f"{value:+.2%}"
            )
        )
    )

print()
print("=" * 150)
print(
    "JANUARY 2026 FORECASTED "
    "AND ACTUAL PRICES"
)
print("=" * 150)

print(
    terminal_table.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 13. Create dark HTML table
# ------------------------------------------------------------

html_columns = [
    "Type",
    "Asset",
    "Ticker",
    "Information date",
    "Actual end date",
    "Actual sessions",
    "Information price",
    "XGBoost predicted price",
    "Sentiment predicted price",
    "Markowitz predicted price",
    "Actual price",
    "XGBoost return",
    "Sentiment return",
    "Markowitz return",
    "Actual return",
    "Sentiment adjustment",
]

styled_table = (
    january_2026_forecast[
        html_columns
    ]
    .style
    .hide(axis="index")
    .format(
        {
            "Actual sessions": "{:d}",
            "Information price": "€{:,.2f}",
            "XGBoost predicted price": "€{:,.2f}",
            "Sentiment predicted price": "€{:,.2f}",
            "Markowitz predicted price": "€{:,.2f}",
            "Actual price": "€{:,.2f}",
            "XGBoost return": "{:+.2%}",
            "Sentiment return": "{:+.2%}",
            "Markowitz return": "{:+.2%}",
            "Actual return": "{:+.2%}",
            "Sentiment adjustment": "{:+.2%}",
        },
        na_rep="—",
    )
    .set_properties(
        **{
            "color": "#ffffff",
            "border-color": "#555b63",
        }
    )
    .set_caption(
        "Forecasted vs Actual Prices and Returns — January 2026"
    )
    .set_table_attributes(
        'style="width:100%;'
        'border-collapse:collapse;'
        'font-family:Arial,sans-serif;'
        'font-size:13px;"'
    )
    .set_table_styles(
        [
            {
                "selector": "caption",
                "props": [
                    ("caption-side", "top"),
                    ("background-color", "#111317"),
                    ("color", "#ffffff"),
                    ("font-size", "16px"),
                    ("font-weight", "700"),
                    ("text-align", "left"),
                    ("padding", "12px"),
                ],
            },
            {
                "selector": "thead th",
                "props": [
                    ("background-color", "#171a1f"),
                    ("color", "#ffffff"),
                    ("font-weight", "600"),
                    ("text-align", "center"),
                    ("padding", "9px"),
                    ("border", "1px solid #555b63"),
                    ("white-space", "nowrap"),
                ],
            },
            {
                "selector": "tbody tr:nth-child(odd) td",
                "props": [
                    ("background-color", "#343a40"),
                    ("color", "#ffffff"),
                ],
            },
            {
                "selector": "tbody tr:nth-child(even) td",
                "props": [
                    ("background-color", "#202328"),
                    ("color", "#ffffff"),
                ],
            },
            {
                "selector": "tbody td",
                "props": [
                    ("padding", "8px 10px"),
                    ("border", "1px solid #555b63"),
                    ("white-space", "nowrap"),
                ],
            },
        ]
    )
)

html_file = os.path.join(
    RESULTS_DIR,
    "january_2026_forecast_table.html",
)

with open(
    html_file,
    "w",
    encoding="utf-8",
) as output:
    output.write(styled_table.to_html())

print()
print(
    f"Dark HTML table saved to: "
    f"{html_file}"
)

# Display the styled table only inside Jupyter.
# This prevents <Styler object ...> from appearing in Terminal.
if "ipykernel" in sys.modules:
    from IPython.display import display
    display(styled_table)


# ------------------------------------------------------------
# 14. Vertical return chart
# ------------------------------------------------------------

return_chart_data = (
    january_2026_forecast
    .set_index("Asset")[
        [
            "XGBoost return",
            "Sentiment return",
            "Markowitz return",
            "Actual return",
        ]
    ]
    * 100.0
)

ax = return_chart_data.plot(
    kind="bar",
    figsize=(14, 7),
    width=0.78,
    color=[
        "#55A868",
        "#C44E52",
        "#4C72B0",
        "#333333",
    ],
    edgecolor="black",
    linewidth=0.5,
)

ax.axhline(
    0,
    color="black",
    linewidth=0.8,
)

ax.set_title(
    "Forecasted vs Actual Returns — January 2026",
    fontsize=14,
    weight="bold",
)

ax.set_xlabel("")
ax.set_ylabel(
    f"{FORECAST_HORIZON}-trading-session return (%)"
)

ax.legend(
    title="Method",
    frameon=False,
    ncol=4,
)

ax.tick_params(
    axis="x",
    rotation=35,
)

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.2f%%",
        padding=3,
        fontsize=8,
        rotation=90,
    )

plt.tight_layout()

return_chart_file = os.path.join(
    RESULTS_DIR,
    "january_2026_returns_chart.png",
)

plt.savefig(
    return_chart_file,
    dpi=200,
    bbox_inches="tight",
)

plt.show()


# ------------------------------------------------------------
# 15. Vertical price chart
# ------------------------------------------------------------

price_chart_data = (
    january_2026_forecast
    .set_index("Asset")[
        [
            "Information price",
            "XGBoost predicted price",
            "Sentiment predicted price",
            "Markowitz predicted price",
            "Actual price",
        ]
    ]
)

ax = price_chart_data.plot(
    kind="bar",
    figsize=(15, 8),
    width=0.82,
    color=[
        "#8C8C8C",
        "#55A868",
        "#C44E52",
        "#4C72B0",
        "#111111",
    ],
    edgecolor="black",
    linewidth=0.5,
)

ax.set_title(
    "Information, Forecasted and Actual Prices — January 2026",
    fontsize=14,
    weight="bold",
)

ax.set_xlabel("")
ax.set_ylabel("Price (€)")

ax.legend(
    title="Price",
    frameon=False,
    ncol=3,
)

ax.tick_params(
    axis="x",
    rotation=35,
)

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="€%.2f",
        padding=3,
        fontsize=7,
        rotation=90,
    )

plt.tight_layout()

price_chart_file = os.path.join(
    RESULTS_DIR,
    "january_2026_vertical_price_chart.png",
)

plt.savefig(
    price_chart_file,
    dpi=200,
    bbox_inches="tight",
)

plt.show()

print()
print(f"Return chart saved to: {return_chart_file}")
print(f"Vertical price chart saved to: {price_chart_file}")