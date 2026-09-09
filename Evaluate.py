"""
evaluate.py

Turns the raw backtest output into the tables and figures that go into
Chapter 5.

Produces:

  performance_metrics.csv   full-sample metrics for all six strategies
  subperiod_metrics.csv     the same metrics inside each market regime
  bootstrap_tests.csv       stationary-bootstrap significance tests
  figures/                  cumulative wealth, drawdown, rolling Sharpe,
                            turnover, feature importance

The significance test is the part that answers the research question. A
higher Sharpe ratio for the sentiment-augmented model means nothing on its
own: with roughly seven years of daily data, differences of 0.2 in Sharpe
arise from noise routinely. The stationary bootstrap of Politis and Romano
(1994) resamples blocks of the paired daily return series, preserving
autocorrelation, and produces a distribution of the Sharpe difference under
resampling. Reporting that distribution is what separates a finding from an
anecdote.

Run after backtest.py.
"""

import os

import numpy as np
import pandas as pd

from data_config import RISKFREE_RATE_FILE
from backtest_config import (
    TRADING_DAYS_PER_YEAR,
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_MEAN_BLOCK,
    BOOTSTRAP_SEED,
    SUBPERIODS,
    RESULTS_DIR,
    FIGURES_DIR,
    PORTFOLIO_RETURNS_FILE,
    TURNOVER_FILE,
    METRICS_FILE,
    SUBPERIOD_METRICS_FILE,
    BOOTSTRAP_FILE,
    FEATURE_IMPORTANCE_FILE,
)


# ---------------------------------------------------------------------------
# Risk-free rate
# ---------------------------------------------------------------------------

def load_daily_risk_free(index):
    """
    Convert the 3-month Euribor series into a daily rate aligned to the
    backtest calendar. The ECB quotes an annual percentage, so it is
    de-annualized geometrically rather than by simple division.

    If the file is missing the function returns zeros and says so, which
    lets the pipeline run end to end before the ECB script has been used.
    """
    if not os.path.exists(RISKFREE_RATE_FILE):
        print(f"NOTE: {RISKFREE_RATE_FILE} not found. Sharpe and Sortino "
              f"ratios will be computed against a zero risk-free rate. "
              f"Run collect_riskfree_rate.py to fix this before writing up.")
        return pd.Series(0.0, index=index)

    rates = pd.read_csv(RISKFREE_RATE_FILE, parse_dates=["date"])
    rates = rates.set_index("date")["euribor_3m_rate"] / 100.0
    daily = (1.0 + rates) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    return daily.reindex(index).ffill().bfill().fillna(0.0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def annualized_return(returns):
    """Geometric, not arithmetic. Arithmetic means overstate compounded
    growth whenever volatility is non-trivial."""
    total = float((1.0 + returns).prod())
    years = len(returns) / TRADING_DAYS_PER_YEAR
    if years <= 0 or total <= 0:
        return np.nan
    return total ** (1.0 / years) - 1.0


def annualized_volatility(returns):
    return float(returns.std(ddof=1)) * np.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(returns, risk_free):
    excess = returns - risk_free.reindex(returns.index).fillna(0.0)
    sigma = float(excess.std(ddof=1))
    if sigma <= 0:
        return np.nan
    return float(excess.mean()) / sigma * np.sqrt(TRADING_DAYS_PER_YEAR)


def sortino_ratio(returns, risk_free):
    """
    Downside deviation uses the full sample in the denominator, not only
    the count of negative days. Dividing by the number of losing days
    inflates the ratio and is a frequent error in applied work.
    """
    excess = returns - risk_free.reindex(returns.index).fillna(0.0)
    downside = excess.clip(upper=0.0)
    downside_deviation = float(np.sqrt((downside ** 2).sum() / len(excess)))
    if downside_deviation <= 0:
        return np.nan
    return float(excess.mean()) / downside_deviation * np.sqrt(TRADING_DAYS_PER_YEAR)


def max_drawdown(returns):
    wealth = (1.0 + returns).cumprod()
    peak = wealth.cummax()
    return float((wealth / peak - 1.0).min())


def drawdown_series(returns):
    wealth = (1.0 + returns).cumprod()
    return wealth / wealth.cummax() - 1.0


def calmar_ratio(returns):
    drawdown = abs(max_drawdown(returns))
    if drawdown <= 0:
        return np.nan
    return annualized_return(returns) / drawdown


def value_at_risk(returns, level=0.05):
    """Historical 5% daily VaR, reported as a positive loss figure."""
    return float(-np.percentile(returns.dropna(), level * 100))


def annualized_turnover(turnover_log, strategy, years):
    subset = turnover_log[turnover_log["strategy"] == strategy]
    if subset.empty or years <= 0:
        return np.nan
    # Exclude the opening trade, which is a one-off entry cost rather than
    # part of the strategy's ongoing trading behaviour.
    ongoing = subset["turnover"].iloc[1:]
    return float(ongoing.sum()) / years


def compute_metrics(returns_frame, risk_free, turnover_log=None):
    years = len(returns_frame) / TRADING_DAYS_PER_YEAR
    rows = []

    for strategy in returns_frame.columns:
        series = returns_frame[strategy].dropna()
        if series.empty:
            continue
        rows.append({
            "strategy": strategy,
            "annual_return": annualized_return(series),
            "annual_volatility": annualized_volatility(series),
            "sharpe": sharpe_ratio(series, risk_free),
            "sortino": sortino_ratio(series, risk_free),
            "max_drawdown": max_drawdown(series),
            "calmar": calmar_ratio(series),
            "var_95_daily": value_at_risk(series),
            "annual_turnover": (annualized_turnover(turnover_log, strategy, years)
                                if turnover_log is not None else np.nan),
            "positive_days_pct": float((series > 0).mean()),
            "trading_days": len(series),
        })

    return pd.DataFrame(rows).set_index("strategy")


# ---------------------------------------------------------------------------
# Stationary bootstrap significance testing
# ---------------------------------------------------------------------------

def stationary_bootstrap_indices(n, mean_block, iterations, rng):
    """
    Politis and Romano (1994), generated for all iterations at once.

    Each position either starts a new block, with probability 1 / mean_block,
    or continues the previous one. Block lengths are therefore geometrically
    distributed with the requested mean, and the series wraps around at the
    end. Random block lengths avoid the sensitivity to one arbitrary fixed
    block length that the moving-block bootstrap suffers from.

    Returns an (iterations, n) integer array. Building it vectorized rather
    than with a nested Python loop is what keeps 5,000 replications over
    eight comparisons down to seconds instead of hours.
    """
    p = 1.0 / mean_block
    positions = np.arange(n)

    restart = rng.random((iterations, n)) < p
    restart[:, 0] = True

    starts = rng.integers(0, n, size=(iterations, n))

    # For each position, find the index of the most recent restart. Marking
    # non-restart positions as -1 and taking a running maximum forward-fills
    # the block origin across the row.
    marked = np.where(restart, positions, -1)
    block_origin = np.maximum.accumulate(marked, axis=1)

    offset = positions - block_origin
    block_seed = np.take_along_axis(starts, block_origin, axis=1)

    return (block_seed + offset) % n


def bootstrap_sharpe_difference(returns_a, returns_b, risk_free,
                                iterations=BOOTSTRAP_ITERATIONS,
                                mean_block=BOOTSTRAP_MEAN_BLOCK,
                                seed=BOOTSTRAP_SEED):
    """
    Two-sided test of the null that two strategies have equal Sharpe ratios.

    Both series are resampled with the SAME block indices at every
    iteration. Resampling them independently would destroy the
    contemporaneous correlation between the strategies, which is very high
    here because they hold overlapping assets, and would badly overstate
    the variance of the difference.

    The p-value is computed by centering the bootstrap distribution on the
    observed difference and asking how often a recentered draw is at least
    as extreme as what was actually observed.
    """
    joined = pd.concat([returns_a, returns_b], axis=1).dropna()
    joined.columns = ["a", "b"]
    rf = risk_free.reindex(joined.index).fillna(0.0)

    excess_a = (joined["a"] - rf).values
    excess_b = (joined["b"] - rf).values
    n = len(joined)

    scale = np.sqrt(TRADING_DAYS_PER_YEAR)

    def sharpe(values, axis=None):
        sigma = values.std(ddof=1, axis=axis)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(sigma > 0,
                            values.mean(axis=axis) / sigma * scale, np.nan)

    observed = float(sharpe(excess_a) - sharpe(excess_b))

    rng = np.random.default_rng(seed)

    # Chunk the replications so memory stays modest on a laptop even with
    # a long sample and a large iteration count.
    chunk = max(1, min(iterations, 20_000_000 // max(n, 1)))
    differences = np.empty(iterations)
    done = 0
    while done < iterations:
        size = min(chunk, iterations - done)
        idx = stationary_bootstrap_indices(n, mean_block, size, rng)
        differences[done:done + size] = (sharpe(excess_a[idx], axis=1)
                                         - sharpe(excess_b[idx], axis=1))
        done += size

    centered = differences - np.nanmean(differences)
    p_value = float(np.mean(np.abs(centered) >= abs(observed)))

    return {
        "sharpe_a": sharpe(excess_a),
        "sharpe_b": sharpe(excess_b),
        "difference": observed,
        "ci_lower_95": float(np.nanpercentile(differences, 2.5)),
        "ci_upper_95": float(np.nanpercentile(differences, 97.5)),
        "p_value": p_value,
        "significant_at_5pct": p_value < 0.05,
        "observations": n,
    }


# The comparisons that matter. The first line is the thesis's primary
# hypothesis test: everything else is context for it.
COMPARISONS = [
    ("xgb_sentiment", "xgb"),
    ("xgb_sentiment", "markowitz"),
    ("xgb_sentiment", "hrp"),
    ("xgb_sentiment", "equal_weight"),
    ("xgb", "markowitz"),
    ("xgb", "hrp"),
    ("xgb", "equal_weight"),
    ("hrp", "markowitz"),
    ("hrp", "equal_weight"),
    ("markowitz", "equal_weight"),
]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(returns_frame, risk_free, turnover_log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9})

    # Cumulative wealth
    fig, ax = plt.subplots(figsize=(9, 5))
    for strategy in returns_frame.columns:
        wealth = (1.0 + returns_frame[strategy].dropna()).cumprod()
        ax.plot(wealth.index, wealth.values, label=strategy, linewidth=1.2)
    ax.set_title("Growth of one euro, net of transaction costs")
    ax.set_ylabel("Cumulative value")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "cumulative_wealth.png"))
    plt.close(fig)

    # Drawdowns
    fig, ax = plt.subplots(figsize=(9, 4))
    for strategy in returns_frame.columns:
        dd = drawdown_series(returns_frame[strategy].dropna())
        ax.plot(dd.index, dd.values * 100, label=strategy, linewidth=1.0)
    ax.set_title("Drawdown from running peak")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "drawdowns.png"))
    plt.close(fig)

    # Rolling one-year Sharpe
    window = TRADING_DAYS_PER_YEAR
    if len(returns_frame) > window:
        fig, ax = plt.subplots(figsize=(9, 4))
        for strategy in returns_frame.columns:
            series = returns_frame[strategy].dropna()
            excess = series - risk_free.reindex(series.index).fillna(0.0)
            rolling = (excess.rolling(window).mean()
                       / excess.rolling(window).std(ddof=1)
                       * np.sqrt(TRADING_DAYS_PER_YEAR))
            ax.plot(rolling.index, rolling.values, label=strategy, linewidth=1.0)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title("Rolling 252-day Sharpe ratio")
        ax.legend(frameon=False, ncol=3, fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "rolling_sharpe.png"))
        plt.close(fig)

    # Turnover per rebalancing
    if turnover_log is not None and not turnover_log.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        for strategy, group in turnover_log.groupby("strategy"):
            ongoing = group.iloc[1:]
            ax.plot(ongoing.index, ongoing["turnover"].values,
                    label=strategy, linewidth=1.0, marker="o", markersize=2)
        ax.set_title("Turnover per rebalancing")
        ax.set_ylabel("Fraction of portfolio traded")
        ax.legend(frameon=False, ncol=3, fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "turnover.png"))
        plt.close(fig)

    # Feature importance
    if os.path.exists(FEATURE_IMPORTANCE_FILE):
        importance = pd.read_csv(FEATURE_IMPORTANCE_FILE)
        sentiment_model = importance[importance["method"] == "xgb_sentiment"]
        if len(sentiment_model):
            averages = (sentiment_model.drop(columns=["date", "method"])
                        .mean().sort_values())
            fig, ax = plt.subplots(figsize=(7, 5))
            colors = ["#c44e52" if "sentiment" in f or "headline" in f
                      else "#4c72b0" for f in averages.index]
            ax.barh(averages.index, averages.values * 100, color=colors)
            ax.set_title("Average XGBoost gain, sentiment-augmented model\n"
                         "(sentiment features in red)")
            ax.set_xlabel("Share of total gain (%)")
            fig.tight_layout()
            fig.savefig(os.path.join(FIGURES_DIR, "feature_importance.png"))
            plt.close(fig)

    print(f"Saved figures to {FIGURES_DIR}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(PORTFOLIO_RETURNS_FILE):
        print(f"ERROR: {PORTFOLIO_RETURNS_FILE} not found. Run backtest.py first.")
        return

    returns_frame = pd.read_csv(PORTFOLIO_RETURNS_FILE,
                                index_col=0, parse_dates=True)
    print(f"Loaded {returns_frame.shape[0]} days x "
          f"{returns_frame.shape[1]} strategies")
    print(f"Out-of-sample period: {returns_frame.index.min().date()} to "
          f"{returns_frame.index.max().date()}")

    risk_free = load_daily_risk_free(returns_frame.index)

    turnover_log = None
    if os.path.exists(TURNOVER_FILE):
        turnover_log = pd.read_csv(TURNOVER_FILE, index_col=0, parse_dates=True)

    # --- Full-sample metrics ---
    metrics = compute_metrics(returns_frame, risk_free, turnover_log)
    metrics.to_csv(METRICS_FILE)

    print("\n" + "=" * 78)
    print("FULL-SAMPLE PERFORMANCE, NET OF TRANSACTION COSTS")
    print("=" * 78)
    display = metrics.copy()
    for col in ["annual_return", "annual_volatility", "max_drawdown",
                "var_95_daily", "positive_days_pct"]:
        display[col] = (display[col] * 100).round(2)
    for col in ["sharpe", "sortino", "calmar", "annual_turnover"]:
        display[col] = display[col].round(3)
    print(display.drop(columns=["trading_days"]).to_string())

    # --- Sub-period metrics ---
    subperiod_rows = []
    for label, (start, end) in SUBPERIODS.items():
        window = returns_frame.loc[start:end]
        if len(window) < 40:
            continue
        sub = compute_metrics(window, risk_free)
        sub["period"] = label
        subperiod_rows.append(sub.reset_index())

    if subperiod_rows:
        subperiods = pd.concat(subperiod_rows, ignore_index=True)
        subperiods.to_csv(SUBPERIOD_METRICS_FILE, index=False)

        print("\n" + "=" * 78)
        print("SHARPE RATIO BY MARKET REGIME")
        print("=" * 78)
        pivot = subperiods.pivot(index="strategy", columns="period",
                                 values="sharpe").round(2)
        print(pivot.to_string())

    # --- Significance tests ---
    print("\n" + "=" * 78)
    print("STATIONARY BOOTSTRAP TESTS OF SHARPE DIFFERENCES")
    print(f"{BOOTSTRAP_ITERATIONS:,} iterations, mean block "
          f"{BOOTSTRAP_MEAN_BLOCK} days")
    print("=" * 78)

    test_rows = []
    for strategy_a, strategy_b in COMPARISONS:
        if strategy_a not in returns_frame or strategy_b not in returns_frame:
            continue
        result = bootstrap_sharpe_difference(returns_frame[strategy_a],
                                             returns_frame[strategy_b],
                                             risk_free)
        result["strategy_a"] = strategy_a
        result["strategy_b"] = strategy_b
        test_rows.append(result)

        print(f"\n{strategy_a} vs {strategy_b}")
        print(f"  Sharpe: {result['sharpe_a']:.3f} vs "
              f"{result['sharpe_b']:.3f}   "
              f"difference {result['difference']:+.3f}")
        print(f"  95% bootstrap interval: "
              f"[{result['ci_lower_95']:+.3f}, {result['ci_upper_95']:+.3f}]")
        print(f"  p-value: {result['p_value']:.4f}   "
              f"{'significant at 5%' if result['significant_at_5pct'] else 'not significant at 5%'}")

    if test_rows:
        tests = pd.DataFrame(test_rows)
        column_order = ["strategy_a", "strategy_b", "sharpe_a", "sharpe_b",
                        "difference", "ci_lower_95", "ci_upper_95",
                        "p_value", "significant_at_5pct", "observations"]
        tests[column_order].to_csv(BOOTSTRAP_FILE, index=False)
        print(f"\nSaved significance tests to {BOOTSTRAP_FILE}")

        primary = tests[(tests["strategy_a"] == "xgb_sentiment")
                        & (tests["strategy_b"] == "xgb")]
        if len(primary):
            row = primary.iloc[0]
            print("\n" + "-" * 78)
            print("PRIMARY RESEARCH QUESTION")
            print("-" * 78)
            print("Does FinBERT sentiment measurably improve risk-adjusted "
                  "performance?")
            verdict = ("Yes, at the 5% level."
                       if row["significant_at_5pct"] and row["difference"] > 0
                       else "No statistically distinguishable improvement.")
            print(f"Sharpe difference {row['difference']:+.3f}, "
                  f"p = {row['p_value']:.4f}. {verdict}")

    make_figures(returns_frame, risk_free, turnover_log)
    print(f"\nAll evaluation output written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()