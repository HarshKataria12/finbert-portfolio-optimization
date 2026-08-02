"""
backtest_config.py

Configuration for the modelling stage of the pipeline (scripts 06-09).

Kept in its own file so data_config.py, which is already finished and
verified for the data-collection stage, does not have to be edited again.
Everything here is imported by 07_forecast_model.py, 08_backtest.py and
09_evaluate.py.
"""

import os

from data_config import DATA_DIR


# ---------------------------------------------------------------------------
# Backtest window
# ---------------------------------------------------------------------------
# Price history starts 2016-01-01. The first three years are consumed by
# the initial training window, so live out-of-sample trading begins in 2019.

BACKTEST_START = "2019-01-01"
BACKTEST_END = "2025-12-31"

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Rebalancing
# ---------------------------------------------------------------------------
# Monthly rebalancing is the default. Week 11's sensitivity analysis
# re-runs the whole backtest with "W-FRI" and "QE" to show how much of the
# result depends on this choice.

REBALANCE_FREQ = "ME"          # pandas offset alias: ME, QE, W-FRI
TRANSACTION_COST = 0.0015      # 0.15% charged on every euro traded

# Partial rebalancing. 1.0 trades all the way to the target weights every
# month. Values below 1.0 blend the new target with the drifted portfolio,
# which cuts turnover at the cost of tracking the model's view more slowly.
# Left off by default; Week 11's sensitivity analysis re-runs with 0.5 and
# 0.25, which matters because the forecast-driven methods turn the whole
# portfolio over roughly once a month at the default setting.
TURNOVER_DAMPING = 1.0


# ---------------------------------------------------------------------------
# Estimation windows
# ---------------------------------------------------------------------------
# COVARIANCE_WINDOW is the rolling window of daily returns used for the
# Ledoit-Wolf covariance estimate and for HRP's correlation clustering.
# 756 trading days is roughly three years, long enough for a stable
# 57 x 57 covariance matrix without reaching too far into stale regimes.

COVARIANCE_WINDOW = 756
MIN_HISTORY = 252              # a ticker needs this much history to be traded


# ---------------------------------------------------------------------------
# Portfolio constraints (retail investor)
# ---------------------------------------------------------------------------
# Long-only, fully invested, no leverage. The position cap prevents the
# optimizer from concentrating everything in one or two assets, which is
# the classic failure mode of unconstrained mean-variance optimization.

MIN_WEIGHT = 0.0
MAX_WEIGHT = 0.15


# ---------------------------------------------------------------------------
# Forecasting model
# ---------------------------------------------------------------------------
# FORECAST_HORIZON: the model predicts the log return over the next 21
# trading days, matching the monthly rebalancing frequency.
#
# EMBARGO_DAYS: no training sample may use a target window that overlaps
# the prediction date. A sample dated t has a target that resolves at
# t + 21, so at a rebalance date T the newest admissible training sample
# is dated T - 21. Without this purge the model trains on information it
# could not have had, which is the single most common source of inflated
# backtest results in this literature.

FORECAST_HORIZON = 21
EMBARGO_DAYS = 21

# Rolling training window for XGBoost, in calendar days. Set to None for
# an expanding window that uses all history up to the embargo cutoff.
TRAIN_WINDOW_DAYS = 1260       # roughly five years

# Retrain every N rebalances. 1 = retrain monthly. Raise to 3 for a
# faster first end-to-end run while debugging.
RETRAIN_EVERY = 1

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "n_jobs": -1,
    "random_state": 42,
    "verbosity": 0,
}


# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------
# The only difference between Method 3 and Method 4 is the four sentiment
# columns. Everything else about the two models is identical, which is what
# makes the comparison a clean ablation.

TECHNICAL_FEATURES = [
    "mom_5d",
    "mom_10d",
    "mom_20d",
    "realized_vol_20d",
    "ma_ratio_50_200",
    "rsi_14d",
    "relative_volume",
]

SENTIMENT_FEATURES = [
    "sentiment_mean",
    "sentiment_std",
    "sentiment_max",
    "headline_count",
]

FEATURE_SETS = {
    "xgb": TECHNICAL_FEATURES,
    "xgb_sentiment": TECHNICAL_FEATURES + SENTIMENT_FEATURES,
}


# ---------------------------------------------------------------------------
# Strategies evaluated
# ---------------------------------------------------------------------------
# The four thesis methods plus two benchmarks required by the proposal.

STRATEGIES = [
    "markowitz",        # Method 1: MVO with Ledoit-Wolf shrinkage
    "hrp",              # Method 2: Hierarchical Risk Parity
    "xgb",              # Method 3: XGBoost forecast into MVO
    "xgb_sentiment",    # Method 4: XGBoost + FinBERT forecast into MVO
    "equal_weight",     # benchmark: 1/N, rebalanced
    "buy_and_hold",     # benchmark: 1/N at inception, never rebalanced
]


# ---------------------------------------------------------------------------
# Bootstrap significance testing
# ---------------------------------------------------------------------------
# Stationary bootstrap (Politis and Romano, 1994). The mean block length
# preserves the autocorrelation structure of daily returns, which a naive
# iid bootstrap would destroy.

BOOTSTRAP_ITERATIONS = 5000
BOOTSTRAP_MEAN_BLOCK = 21
BOOTSTRAP_SEED = 42


# ---------------------------------------------------------------------------
# Sub-periods for regime robustness (Week 10)
# ---------------------------------------------------------------------------

SUBPERIODS = {
    "2019 calm bull": ("2019-01-01", "2019-12-31"),
    "2020 covid crash and recovery": ("2020-01-01", "2020-12-31"),
    "2021 recovery bull": ("2021-01-01", "2021-12-31"),
    "2022 inflation bear": ("2022-01-01", "2022-12-31"),
    "2023-2025 late sample": ("2023-01-01", "2025-12-31"),
}


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

RESULTS_DIR = os.path.join(DATA_DIR, "results")

PANEL_FILE = os.path.join(RESULTS_DIR, "model_panel.csv")
WEIGHTS_FILE = os.path.join(RESULTS_DIR, "weights_history.csv")
PORTFOLIO_RETURNS_FILE = os.path.join(RESULTS_DIR, "portfolio_returns.csv")
TURNOVER_FILE = os.path.join(RESULTS_DIR, "turnover_history.csv")
METRICS_FILE = os.path.join(RESULTS_DIR, "performance_metrics.csv")
SUBPERIOD_METRICS_FILE = os.path.join(RESULTS_DIR, "subperiod_metrics.csv")
BOOTSTRAP_FILE = os.path.join(RESULTS_DIR, "bootstrap_tests.csv")
FEATURE_IMPORTANCE_FILE = os.path.join(RESULTS_DIR, "feature_importance.csv")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
