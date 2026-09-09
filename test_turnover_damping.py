"""
test_turnover_damping.py

Quick, no-retraining test of how much turnover damping changes strategy
performance. Reuses the already-fitted portfolio weights in
data/results/weights_history.csv (produced by a previous run of
Backtest.py) and re-simulates them at several TURNOVER_DAMPING values,
without refitting any XGBoost models. This is the exact logic used to
produce the 0.242 -> 0.625 Sharpe numbers discussed earlier.

Usage (from inside the model/ folder):
    python3 test_turnover_damping.py

Requirement: data/results/weights_history.csv must already exist. If it
doesn't, run `python3 Backtest.py` once first (with TURNOVER_DAMPING left
at whatever value backtest_config.py currently has) -- that produces the
weights this script re-simulates. You only need to do that once, even if
you then want to try several different damping values afterwards.

Output: prints a summary table, and saves it to
data/results/turnover_damping_sensitivity.csv
"""
import os
import sys

sys.path.insert(0, ".")

import pandas as pd

from Backtest import load_returns, simulate
from Evaluate import load_daily_risk_free, compute_metrics
from backtest_config import RESULTS_DIR, WEIGHTS_FILE

# Edit this list to try other damping values.
DAMPING_VALUES = [1.0, 0.5, 0.3, 0.15]

# Edit this list to include/exclude strategies from the comparison.
STRATEGIES_TO_TEST = ["xgb", "xgb_sentiment", "markowitz", "hrp", "equal_weight"]


def load_schedules(weights_file=WEIGHTS_FILE):
    if not os.path.exists(weights_file):
        raise FileNotFoundError(
            f"{weights_file} not found.\n\n"
            f"Run `python3 Backtest.py` once first (with TURNOVER_DAMPING "
            f"left at its normal value in backtest_config.py) to produce "
            f"it. This script only re-simulates the weights it already "
            f"contains at different damping levels -- it does not refit "
            f"any XGBoost models, so it runs in seconds rather than "
            f"minutes."
        )
    wh = pd.read_csv(weights_file, parse_dates=["date"])
    schedules = {}
    for strat, g in wh.groupby("strategy"):
        sched = {}
        for date, gg in g.groupby("date"):
            sched[date] = pd.Series(gg["weight"].values, index=gg["ticker"].values)
        schedules[strat] = sched
    return schedules


def main():
    prices, returns = load_returns()
    all_tickers = list(returns.columns)
    schedules = load_schedules()
    risk_free = load_daily_risk_free(returns.index)

    available = [s for s in STRATEGIES_TO_TEST if s in schedules]
    missing = [s for s in STRATEGIES_TO_TEST if s not in schedules]
    if missing:
        print(f"NOTE: {missing} not found in {WEIGHTS_FILE}, skipping.")

    rows = []
    for damping in DAMPING_VALUES:
        frame = {}
        for strat in available:
            series, _ = simulate(returns, schedules[strat], all_tickers,
                                 damping=damping)
            frame[strat] = series
        frame = pd.DataFrame(frame).dropna(how="all")
        metrics = compute_metrics(frame, risk_free, None).reset_index()
        metrics.insert(0, "damping", damping)
        rows.append(metrics)

    summary = pd.concat(rows, ignore_index=True)
    display_cols = ["damping", "strategy", "annual_return",
                    "annual_volatility", "sharpe", "max_drawdown"]
    print("\n" + "=" * 78)
    print("TURNOVER DAMPING SENSITIVITY (same forecasts, no refitting)")
    print("=" * 78)
    print(summary[display_cols].round(3).to_string(index=False))

    out_path = os.path.join(RESULTS_DIR, "turnover_damping_sensitivity.csv")
    summary.to_csv(out_path, index=False)
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()
