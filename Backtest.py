"""
backtest.py

Walk-forward, out-of-sample backtest of all six strategies.

At every rebalancing date the engine:

  1. Freezes an information set consisting only of data available on or
     before that date's close.
  2. Estimates a Ledoit-Wolf covariance matrix from the trailing window.
  3. Constructs six weight vectors: the four thesis methods plus the
     equal-weight and buy-and-hold benchmarks.
  4. Charges transaction costs on the difference between the new target
     weights and the weights that drifted in since the last rebalancing.

Between rebalancing dates, weights drift with realized returns rather than
being silently held constant. Ignoring drift is a common shortcut that
understates turnover and therefore overstates net performance.

Ordering convention, which matters for look-ahead:
  weights decided at the close of date t are earned on the returns of
  t + 1 onwards. Nothing computed on date t is ever used to earn a return
  on date t.

Outputs, all under data/results/:
  portfolio_returns.csv   daily net returns per strategy
  weights_history.csv     the weight vector chosen at every rebalancing
  turnover_history.csv    per-rebalancing turnover and cost per strategy
  feature_importance.csv  average XGBoost gain per feature per method

Run after build_features.py. Expect roughly five to fifteen minutes on a
MacBook Air, almost all of it spent refitting XGBoost.
"""

import os
import time

import numpy as np
import pandas as pd

from data_config import CLEAN_PRICES_FILE
from backtest_config import (
    BACKTEST_START,
    BACKTEST_END,
    REBALANCE_FREQ,
    TRANSACTION_COST,
    TURNOVER_DAMPING,
    COVARIANCE_WINDOW,
    MIN_HISTORY,
    MAX_WEIGHT,
    FEATURE_SETS,
    RETRAIN_EVERY,
    STRATEGIES,
    RESULTS_DIR,
    WEIGHTS_FILE,
    PORTFOLIO_RETURNS_FILE,
    TURNOVER_FILE,
    FEATURE_IMPORTANCE_FILE,
)
from optimizer import (
    ledoit_wolf_covariance,
    historical_mean_returns,
    mean_variance_weights,
    minimum_variance_weights,
    hrp_weights,
    equal_weights,
)
from forecast_model import (
    load_panel,
    ReturnForecaster,
    annualize_forecast,
)


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def load_returns(prices_file=CLEAN_PRICES_FILE):
    """Daily simple returns. Simple rather than log because portfolio
    returns aggregate linearly across assets only in simple form."""
    prices = pd.read_csv(prices_file, index_col=0, parse_dates=True)
    prices = prices.sort_index()
    returns = prices.pct_change()
    return prices, returns


def get_rebalance_dates(trading_dates, start, end, freq=REBALANCE_FREQ):
    """
    The last actual trading day of each period in the backtest window.
    Using real trading days rather than calendar month-ends avoids
    scheduling a rebalancing on a day the exchange was closed.
    """
    dates = pd.DatetimeIndex(trading_dates)
    dates = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    grouped = pd.Series(dates, index=dates).resample(freq).last().dropna()
    return pd.DatetimeIndex(grouped.values)


def eligible_universe(returns, as_of, window=COVARIANCE_WINDOW,
                      min_history=MIN_HISTORY):
    """
    Assets tradeable at `as_of`: enough history, no gaps in the estimation
    window, and non-zero variance. Screening here rather than dropping
    assets globally means a ticker that lists mid-sample simply enters the
    universe when it becomes available, which is what a real investor would
    experience.
    """
    history = returns.loc[:as_of]
    if len(history) < min_history:
        return [], history

    window_slice = history.tail(window)
    complete = window_slice.columns[window_slice.notna().all()]
    non_degenerate = [t for t in complete if window_slice[t].std() > 1e-12]
    return list(non_degenerate), window_slice


# ---------------------------------------------------------------------------
# Weight construction at one rebalancing date
# ---------------------------------------------------------------------------

def build_weights_for_date(as_of, returns, panel, trading_dates,
                           forecasters, refit=True):
    """
    Returns a dict mapping strategy name to a weight Series, plus a dict of
    diagnostic notes. buy_and_hold is handled by the simulator, not here,
    because it only ever trades once.
    """
    universe, window_slice = eligible_universe(returns, as_of)
    notes = {"universe_size": len(universe)}

    if len(universe) < 5:
        return {}, notes

    lookback = window_slice[universe]
    covariance = ledoit_wolf_covariance(lookback)
    mu_historical = historical_mean_returns(lookback)

    weights = {}

    # Method 1: classical mean-variance with shrunk covariance
    weights["markowitz"] = mean_variance_weights(mu_historical, covariance)

# Method 2: hierarchical risk parity
    weights["hrp"] = hrp_weights(lookback)

    # Benchmark: minimum-variance, ignores expected returns entirely and
    # optimises purely on the shrunk covariance matrix
    weights["minimum_variance"] = minimum_variance_weights(covariance)

    # Methods 3 and 4: XGBoost forecast feeding the same optimizer
    for name in ("xgb", "xgb_sentiment"):
        forecaster = forecasters[name]
        raw = forecaster.fit_predict(panel, trading_dates, as_of,
                                     universe=universe, refit=refit)
        if raw is None or raw.empty:
            # Not enough training history yet. Falling back to the
            # historical mean keeps the strategy invested rather than
            # silently dropping it from the comparison.
            weights[name] = mean_variance_weights(mu_historical, covariance)
            notes[f"{name}_fallback"] = True
            continue

        raw = raw.reindex(universe).dropna()
        mu_predicted = annualize_forecast(raw)
        weights[name] = mean_variance_weights(mu_predicted,
                                              covariance.loc[mu_predicted.index,
                                                             mu_predicted.index])

    # Benchmark: equal weight over the same eligible universe
    weights["equal_weight"] = equal_weights(universe)

    return weights, notes


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(returns, weight_schedule, all_tickers,
             transaction_cost=TRANSACTION_COST, buy_and_hold=False,
             damping=TURNOVER_DAMPING):
    """
    Walk the daily return series forward, drifting weights between
    rebalancings and charging costs on realized trades.

    Parameters
    ----------
    weight_schedule : dict of {rebalance_date: Series of target weights}
    buy_and_hold : if True, only the first entry in the schedule is traded
        and the portfolio drifts untouched thereafter

    Returns
    -------
    daily_returns : Series of net portfolio returns
    turnover_log  : DataFrame of date, turnover, cost
    """
    schedule_dates = sorted(weight_schedule.keys())
    if not schedule_dates:
        return pd.Series(dtype=float), pd.DataFrame()

    first = schedule_dates[0]
    sim_dates = returns.index[returns.index > first]

    holdings = pd.Series(0.0, index=all_tickers)
    initial = weight_schedule[first].reindex(all_tickers).fillna(0.0)
    holdings = initial.copy()

    # The opening trade moves from cash into the first target portfolio, so
    # it costs the full notional. Charging it keeps every strategy on equal
    # footing, including buy-and-hold.
    opening_cost = float(initial.abs().sum()) * transaction_cost

    net_returns = {}
    turnover_rows = [{"date": first, "turnover": float(initial.abs().sum()),
                      "cost": opening_cost}]
    pending_cost = opening_cost

    rebalance_set = set(schedule_dates[1:]) if not buy_and_hold else set()

    for day in sim_dates:
        day_returns = returns.loc[day].reindex(all_tickers).fillna(0.0)

        gross = float((holdings * day_returns).sum())

        # Drift: yesterday's weights grow with today's returns and are
        # renormalized so they continue to sum to one.
        grown = holdings * (1.0 + day_returns)
        total = grown.sum()
        holdings = grown / total if total > 0 else holdings

        cost_today = pending_cost
        pending_cost = 0.0

        if day in rebalance_set:
            is_final_day = (day == sim_dates[-1])
            if is_final_day:
                # No subsequent trading day exists to book this cost
                # against, and no future return would ever use these new
                # weights. Skip constructing a rebalance nobody could
                # ever trade on, rather than logging a cost that then
                # silently never gets deducted from anything.
                print(f"  Skipping rebalance on {day.date()}: "
                      f"no subsequent trading day to apply it to.")
            else:
                target = weight_schedule[day].reindex(all_tickers).fillna(0.0)
                if damping < 1.0:
                    target = damping * target + (1.0 - damping) * holdings
                    total_target = target.sum()
                    if total_target > 0:
                        target = target / total_target
                traded = float((target - holdings).abs().sum())
                # Costs are incurred at the close of the rebalancing day and
                # are booked against the following day's return, matching the
                # convention that new weights only earn from t + 1.
                pending_cost = traded * transaction_cost
                turnover_rows.append({"date": day, "turnover": traded,
                                      "cost": pending_cost})
                holdings = target

        net_returns[day] = gross - cost_today

    return (pd.Series(net_returns).sort_index(),
            pd.DataFrame(turnover_rows).set_index("date"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    started = time.time()

    prices, returns = load_returns()
    print(f"Loaded prices: {prices.shape[0]} days x {prices.shape[1]} assets")
    print(f"Full sample: {prices.index.min().date()} to "
          f"{prices.index.max().date()}")

    panel = load_panel()
    trading_dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    print(f"Loaded model panel: {len(panel):,} rows")

    rebalance_dates = get_rebalance_dates(returns.index,
                                          BACKTEST_START, BACKTEST_END)
    print(f"\nRebalancing on {len(rebalance_dates)} dates, "
          f"{rebalance_dates[0].date()} to {rebalance_dates[-1].date()}")
    print(f"Frequency: {REBALANCE_FREQ}, "
          f"transaction cost: {TRANSACTION_COST * 100:.2f}% per euro traded")
    print(f"Constraints: long only, fully invested, "
          f"position cap {MAX_WEIGHT * 100:.0f}%")

    forecasters = {name: ReturnForecaster(cols)
                   for name, cols in FEATURE_SETS.items()}

    schedules = {name: {} for name in STRATEGIES if name != "buy_and_hold"}
    importance_rows = []
    weight_rows = []

    print("\n--- Walk-forward weight construction ---")
    for step, as_of in enumerate(rebalance_dates):
        refit = (step % RETRAIN_EVERY == 0)
        weights, notes = build_weights_for_date(
            as_of, returns, panel, trading_dates, forecasters, refit=refit)

        if not weights:
            print(f"{as_of.date()}  skipped, universe too small "
                  f"({notes.get('universe_size', 0)} assets)")
            continue

        for name, w in weights.items():
            schedules[name][as_of] = w
            for ticker, value in w[w > 0].items():
                weight_rows.append({"date": as_of, "strategy": name,
                                    "ticker": ticker, "weight": float(value)})

        if refit:
            for name, forecaster in forecasters.items():
                gains = forecaster.feature_importance()
                if gains is not None:
                    row = {"date": as_of, "method": name}
                    row.update(gains.to_dict())
                    importance_rows.append(row)

        if step % 6 == 0 or step == len(rebalance_dates) - 1:
            active = {k: int((v > 0).sum()) for k, v in weights.items()}
            print(f"{as_of.date()}  universe={notes['universe_size']:2d}  "
                  f"holdings: " +
                  "  ".join(f"{k}={v}" for k, v in active.items()))

    print(f"\nWeight construction finished in {time.time() - started:.0f}s")

    # --- Simulate every strategy on the same price series ---
    print("\n--- Simulating net-of-cost daily returns ---")
    all_tickers = list(returns.columns)
    portfolio_returns = {}
    turnover_logs = []

    for name in STRATEGIES:
        if name == "buy_and_hold":
            # 1/N at the first rebalancing date, never touched again.
            source = schedules["equal_weight"]
            if not source:
                continue
            first = sorted(source.keys())[0]
            schedule = {first: source[first]}
            series, log = simulate(returns, schedule, all_tickers,
                                   buy_and_hold=True)
        else:
            series, log = simulate(returns, schedules[name], all_tickers)

        portfolio_returns[name] = series
        log = log.copy()
        log["strategy"] = name
        turnover_logs.append(log)
        print(f"{name:<15} {len(series)} trading days simulated")

    returns_frame = pd.DataFrame(portfolio_returns).dropna(how="all")
    returns_frame.index.name = "date"
    returns_frame.to_csv(PORTFOLIO_RETURNS_FILE)
    print(f"\nSaved daily portfolio returns to {PORTFOLIO_RETURNS_FILE}")

    pd.DataFrame(weight_rows).to_csv(WEIGHTS_FILE, index=False)
    print(f"Saved weight history to {WEIGHTS_FILE}")

    pd.concat(turnover_logs).to_csv(TURNOVER_FILE)
    print(f"Saved turnover history to {TURNOVER_FILE}")

    if importance_rows:
        importance = pd.DataFrame(importance_rows)
        importance.to_csv(FEATURE_IMPORTANCE_FILE, index=False)
        print(f"Saved feature importances to {FEATURE_IMPORTANCE_FILE}")

        print("\n--- Average feature importance, sentiment-augmented model ---")
        sentiment_rows = importance[importance["method"] == "xgb_sentiment"]
        if len(sentiment_rows):
            averages = (sentiment_rows.drop(columns=["date", "method"])
                        .mean().sort_values(ascending=False))
            for feature, value in averages.items():
                print(f"  {feature:<22} {value * 100:5.2f}%")

    print(f"\nTotal runtime: {time.time() - started:.0f}s")
    print("Next: run evaluate.py to produce the metrics tables and figures.")


if __name__ == "__main__":
    main()