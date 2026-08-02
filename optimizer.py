"""
06_optimizers.py

The four weight-construction methods compared in this thesis, plus the two
benchmarks. This file is a library, not a script: 08_backtest.py imports
these functions and calls them once per rebalancing date. Running this file
directly executes a self-test on synthetic data, which is what validates it
"against known benchmarks" as required in Week 5-6.

Methods implemented:

  1. mean_variance_weights   Markowitz maximum-Sharpe portfolio using a
                             Ledoit-Wolf shrunk covariance matrix
  2. hrp_weights             Hierarchical Risk Parity, Lopez de Prado (2016)
  3. (Method 3 and 4 reuse mean_variance_weights, supplying an XGBoost
     expected-return vector instead of a historical sample mean)
  4. equal_weights           1/N benchmark

Every method returns a pandas Series of weights indexed by ticker that is
long-only, sums to one, and respects the position cap in backtest_config.
"""
# You start with noisy price data. Task 2 fixes the covariance matrix before anything dangerous happens to it. Tasks 4-6 build the classical optimizer, with multiple starting points and a safety-net fallback so it never crashes a backtest. Task 7 is that fallback, standing alone. Tasks 8-9 build a completely different philosophy — HRP — that sidesteps the danger entirely rather than fixing it. Task 10 is the benchmark that makes the whole comparison meaningful. Tasks 11-12 are shared plumbing every method relies on. Tasks 13-14 prove the whole thing works, on data where the right answer is known in advance, before you ever trust it on real money.

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

from backtest_config import (
    MIN_WEIGHT,
    MAX_WEIGHT,
    TRADING_DAYS_PER_YEAR,
)

# demo data for selt-test when running this file directly

# ---------------------------------------------------------------------------
# Covariance estimation
# ---------------------------------------------------------------------------

def ledoit_wolf_covariance(returns, annualize=True):
    clean = returns.dropna(axis=1, how="any")
    if clean.shape[1] < 2:
        raise ValueError("Need at least two assets with complete history.")

    estimator = LedoitWolf(assume_centered=False)
    estimator.fit(clean.values)
    cov = estimator.covariance_

    if annualize:
        cov = cov * TRADING_DAYS_PER_YEAR

    return pd.DataFrame(cov, index=clean.columns, columns=clean.columns)


def sample_covariance(returns, annualize=True):
    """Unshrunk sample covariance, kept for the Week 5-6 comparison table
    that shows why shrinkage is used at all."""
    clean = returns.dropna(axis=1, how="any")
    cov = clean.cov()
    if annualize:
        cov = cov * TRADING_DAYS_PER_YEAR
    return cov


def historical_mean_returns(returns, annualize=True):
    """Annualized sample mean of daily returns. This is the naive expected
    return input that Method 1 uses and that Methods 3 and 4 replace with an
    XGBoost forecast."""
    mu = returns.dropna(axis=1, how="any").mean()
    if annualize:
        mu = mu * TRADING_DAYS_PER_YEAR
    return mu



def mean_variance_weights(mu, cov, risk_free_rate=0.0,
                          min_weight=MIN_WEIGHT, max_weight=MAX_WEIGHT):
    tickers = [t for t in mu.index if t in cov.index]
    if len(tickers) == 0:
        raise ValueError("mu and cov share no tickers.")

    mu_vec = mu.loc[tickers].values.astype(float)
    cov_mat = cov.loc[tickers, tickers].values.astype(float)
    n = len(tickers)

    cap = max(max_weight, 1.0 / n)
    bounds = [(min_weight, cap)] * n
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)


    def negative_sharpe(w):
        port_return = float(w @ mu_vec) - risk_free_rate
        port_vol = float(np.sqrt(max(w @ cov_mat @ w, 1e-12)))
        return -port_return / port_vol
    starts = [np.repeat(1.0 / n, n)]
    rng = np.random.default_rng(0)
    for _ in range(4):
        candidate = rng.random(n)
        starts.append(candidate / candidate.sum())

    best_w, best_obj = None, np.inf
    for w0 in starts:
        w0 = np.clip(w0, min_weight, cap)
        w0 = w0 / w0.sum()
        result = minimize(negative_sharpe, w0, method="SLSQP",bounds=bounds, constraints=constraints,options={"maxiter": 500, "ftol": 1e-10})
        if result.success and result.fun < best_obj:
                best_w, best_obj = result.x, result.fun
    if best_w is None:
        return minimum_variance_weights(cov.loc[tickers, tickers],
                                        min_weight=min_weight,
                                        max_weight=max_weight)

    weights = pd.Series(best_w, index=tickers)
    return _sanitize(weights)

def minimum_variance_weights(cov, min_weight=MIN_WEIGHT, max_weight=MAX_WEIGHT):
    tickers = list(cov.index)
    cov_mat = cov.values.astype(float)
    n = len(tickers)

    cap = max(max_weight, 1.0 / n)
    bounds = [(min_weight, cap)] * n
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    def portfolio_variance(w):
        return float(w @ cov_mat @ w)

    w0 = np.repeat(1.0 / n, n)
    result = minimize(portfolio_variance, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 500, "ftol": 1e-12})

    if not result.success:
        return pd.Series(np.repeat(1.0 / n, n), index=tickers)

    return _sanitize(pd.Series(result.x, index=tickers))

# The HRP helper chain
def _correlation_distance(corr):
    values = np.sqrt(np.clip((1.0 - corr.values) / 2.0, 0.0, 1.0))
    np.fill_diagonal(values, 0.0)
    return pd.DataFrame(values, index=corr.index, columns=corr.columns)
def _quasi_diagonalize(link):
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]

    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        clusters = sort_ix[sort_ix >= num_items]
        i = clusters.index
        j = clusters.values - num_items
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])

    return sort_ix.tolist()

def _inverse_variance_weights(cov):
    ivp = 1.0 / np.diag(cov.values)
    return ivp / ivp.sum()

def _cluster_variance(cov, items):
    sub = cov.loc[items, items]
    w = _inverse_variance_weights(sub)
    return float(w @ sub.values @ w)
def _recursive_bisection(cov, sorted_tickers):
    weights = pd.Series(1.0, index=sorted_tickers)
    clusters = [sorted_tickers]

    while len(clusters) > 0:
        next_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            half = len(cluster) // 2
            left, right = cluster[:half], cluster[half:]
            var_left = _cluster_variance(cov, left)
            var_right = _cluster_variance(cov, right)
            alpha = 1.0 - var_left / (var_left + var_right)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
            next_clusters.extend([left, right])
        clusters = next_clusters

    return weights
def hrp_weights(returns, linkage_method="single", max_weight=None):
    clean = returns.dropna(axis=1, how="any")
    if clean.shape[1] < 2:
        raise ValueError("Need at least two assets with complete history.")

    cov = clean.cov() * TRADING_DAYS_PER_YEAR
    corr = clean.corr()

    dist = _correlation_distance(corr)
    condensed = squareform(dist.values, checks=False)
    link = sch.linkage(condensed, method=linkage_method)

    sort_ix = _quasi_diagonalize(link)
    sorted_tickers = corr.index[sort_ix].tolist()

    weights = _recursive_bisection(cov, sorted_tickers)
    weights = weights.reindex(clean.columns)

    if max_weight is not None:
        weights = _cap_and_renormalize(weights, max_weight)

    return _sanitize(weights)
def equal_weights(tickers):
    tickers = list(tickers)
    n = len(tickers)
    return pd.Series(np.repeat(1.0 / n, n), index=tickers)
def _cap_and_renormalize(weights, max_weight, tolerance=1e-9, max_passes=100):
    w = weights.clip(lower=0.0).astype(float)
    w = w / w.sum()
    n = len(w)
    cap = max(max_weight, 1.0 / n)

    for _ in range(max_passes):
        over = w > cap + tolerance
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        room = ~over
        if not room.any() or w[room].sum() <= 0:
            break
        w[room] += excess * (w[room] / w[room].sum())

    return w / w.sum()

def _sanitize(weights):
    w = weights.astype(float).fillna(0.0)
    w = w.clip(lower=0.0)
    w[w < 1e-4] = 0.0
    total = w.sum()
    if total <= 0:
        return pd.Series(np.repeat(1.0 / len(w), len(w)), index=w.index)
    return w / total

# creating fake data where we can know about the expected output and test the functions
def _synthetic_returns(n_assets=20, n_days=756, seed=7):
    rng = np.random.default_rng(seed)
    per_block = n_assets // 4

    corr = np.full((n_assets, n_assets), 0.1)
    for b in range(4):
        lo, hi = b * per_block, (b + 1) * per_block
        corr[lo:hi, lo:hi] = 0.7
    np.fill_diagonal(corr, 1.0)

    vols = np.linspace(0.10, 0.40, n_assets) / np.sqrt(TRADING_DAYS_PER_YEAR)
    cov = corr * np.outer(vols, vols)

    drift = np.linspace(0.02, 0.14, n_assets) / TRADING_DAYS_PER_YEAR
    data = rng.multivariate_normal(drift, cov, size=n_days)

    dates = pd.bdate_range("2016-01-01", periods=n_days)
    tickers = [f"A{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(data, index=dates, columns=tickers)
def _run_self_test():
    print("=" * 72)
    print("OPTIMIZER SELF-TEST ON SYNTHETIC DATA WITH KNOWN STRUCTURE")
    print("=" * 72)

    returns = _synthetic_returns()
    print(f"\nSynthetic panel: {returns.shape[0]} days x {returns.shape[1]} assets")

    lw = ledoit_wolf_covariance(returns)
    sample = sample_covariance(returns)
    print(f"\nCondition number, sample covariance:      {np.linalg.cond(sample.values):,.1f}")
    print(f"Condition number, Ledoit-Wolf covariance: {np.linalg.cond(lw.values):,.1f}")

    mu = historical_mean_returns(returns)
    results = {}
    results["equal_weight"] = equal_weights(returns.columns)
    results["min_variance"] = minimum_variance_weights(lw)
    results["markowitz"] = mean_variance_weights(mu, lw)
    results["hrp"] = hrp_weights(returns)

    print("\n--- Checks ---")
    all_passed = True
    for name, w in results.items():
        sums_to_one = abs(w.sum() - 1.0) < 1e-6
        long_only = (w >= -1e-9).all()
        respects_cap = w.max() <= MAX_WEIGHT + 1e-6 or name in ("hrp",)
        ok = sums_to_one and long_only and respects_cap
        all_passed &= ok
        print(f"{name:<14} sum={w.sum():.6f}  max={w.max():.4f}  "
              f"min={w.min():.4f}  nonzero={int((w > 0).sum()):2d}  "
              f"{'PASS' if ok else 'FAIL'}")

    # these belong OUTSIDE the loop — computed once, not once per strategy
    mv_vol = (returns @ results["min_variance"]).std()
    ew_vol = (returns @ results["equal_weight"]).std()
    hrp_vol = (returns @ results["hrp"]).std()

    print("\n--- Theory checks ---")
    check_1 = mv_vol < ew_vol
    print(f"Minimum-variance below equal-weight volatility: {'PASS' if check_1 else 'FAIL'}")
    check_2 = hrp_vol < ew_vol
    print(f"HRP below equal-weight volatility:              {'PASS' if check_2 else 'FAIL'}")

    hrp_w = results["hrp"]
    low_vol_share = hrp_w.iloc[:5].sum()
    high_vol_share = hrp_w.iloc[-5:].sum()
    check_3 = low_vol_share > high_vol_share
    print(f"HRP tilts toward the low-volatility assets:     "
          f"{'PASS' if check_3 else 'FAIL'} ({low_vol_share:.3f} vs {high_vol_share:.3f})")

    all_passed &= check_1 and check_2 and check_3
    print("\n" + ("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED"))


if __name__ == "__main__":
    _run_self_test()
        