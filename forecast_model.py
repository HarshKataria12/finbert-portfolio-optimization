import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from data_config import CLEAN_PRICES_FILE, FEATURES_FILE
from backtest_config import (
    FORECAST_HORIZON,
    EMBARGO_DAYS,
    TRAIN_WINDOW_DAYS,
    XGB_PARAMS,
    FEATURE_SETS,
    TRADING_DAYS_PER_YEAR,
    PANEL_FILE,
    RESULTS_DIR,
)
def build_targets(prices, horizon=FORECAST_HORIZON):
    forward = np.log(prices.shift(-horizon) / prices)

    long = forward.stack(future_stack=True).reset_index()
    long.columns = ["date", "ticker", "target"]
    return long
def load_panel(features_file=FEATURES_FILE, prices_file=CLEAN_PRICES_FILE,
               horizon=FORECAST_HORIZON):
    if not os.path.exists(features_file):
        raise FileNotFoundError(...)
    if not os.path.exists(prices_file):
        raise FileNotFoundError(...)

    features = pd.read_csv(features_file, parse_dates=["date"])
    prices = pd.read_csv(prices_file, index_col=0, parse_dates=True)

    targets = build_targets(prices, horizon)

    for frame in (features, targets):
        if frame["date"].dt.tz is not None:
            frame["date"] = (frame["date"].dt.tz_convert("UTC")
                             .dt.tz_localize(None))

    panel = features.merge(targets, on=["date", "ticker"], how="left")
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    return panel
def training_cutoff(trading_dates, rebalance_date, embargo=EMBARGO_DAYS):
    position = trading_dates.searchsorted(pd.Timestamp(rebalance_date),
                                          side="right") - 1
    cutoff_position = position - embargo
    if cutoff_position < 0:
        return None
    return trading_dates[cutoff_position]
class ReturnForecaster:
    def __init__(self, feature_cols, params=None,
                 train_window_days=TRAIN_WINDOW_DAYS,
                 embargo=EMBARGO_DAYS,
                 horizon=FORECAST_HORIZON):
        self.feature_cols = list(feature_cols)
        self.params = dict(params or XGB_PARAMS)
        self.train_window_days = train_window_days
        self.embargo = embargo
        self.horizon = horizon
        self.model = None
        self.last_train_rows = 0
        self.last_train_end = None

def _training_slice(self, panel, trading_dates, rebalance_date):
        cutoff = training_cutoff(trading_dates, rebalance_date, self.embargo)
        if cutoff is None:
            return None, None

        mask = panel["date"] <= cutoff
        if self.train_window_days is not None:
            window_start = cutoff - pd.Timedelta(days=self.train_window_days)
            mask &= panel["date"] >= window_start

        train = panel.loc[mask].dropna(subset=self.feature_cols + ["target"])
        return train, cutoff
def fit_predict(self, panel, trading_dates, rebalance_date,
                    universe=None, refit=True):
        predict_slice = panel[panel["date"] == pd.Timestamp(rebalance_date)]
        predict_slice = predict_slice.dropna(subset=self.feature_cols)
        if universe is not None:
            predict_slice = predict_slice[predict_slice["ticker"].isin(universe)]
        if predict_slice.empty:
            return None
        if refit or self.model is None:
            train, cutoff = self._training_slice(panel, trading_dates, rebalance_date)
            if train is None or len(train) < 500:
                return None

            train_x, predict_x = zscore_with_training_stats(
                train[self.feature_cols], predict_slice[self.feature_cols])

            self.model = XGBRegressor(**self.params)
            self.model.fit(train_x.values, train["target"].values)
            self.last_train_rows = len(train)
            self.last_train_end = cutoff
            self._last_mean = train[self.feature_cols].mean()
            self._last_std = (train[self.feature_cols].std()
                              .replace(0.0, np.nan).fillna(1.0))
        else:
            predict_x = ((predict_slice[self.feature_cols] - self._last_mean)
                         / self._last_std)
            predictions = self.model.predict(predict_x.values)
        return pd.Series(predictions, index=predict_slice["ticker"].values)
def feature_importance(self):
        if self.model is None:
            return None
        booster = self.model.get_booster()
        gains = booster.get_score(importance_type="gain")
        mapped = {self.feature_cols[int(k[1:])]: v for k, v in gains.items()}
        series = pd.Series(mapped).reindex(self.feature_cols).fillna(0.0)
        total = series.sum()
        return series / total if total > 0 else series
def annualize_forecast(log_return_forecast, horizon=FORECAST_HORIZON):
    simple = np.exp(log_return_forecast) - 1.0
    annual = (1.0 + simple) ** (TRADING_DAYS_PER_YEAR / horizon) - 1.0

    lower = annual.quantile(0.05)
    upper = annual.quantile(0.95)
    return annual.clip(lower=lower, upper=upper)

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    panel = load_panel()
    print(f"Panel built: {len(panel):,} ticker-day rows, "
          f"{panel['ticker'].nunique()} tickers")
    prices = pd.read_csv(CLEAN_PRICES_FILE, index_col=0, parse_dates=True)
    ticker = prices.columns[0]
    probe_date = prices.index[500]
    expected = np.log(prices[ticker].iloc[500 + FORECAST_HORIZON]
                      / prices[ticker].iloc[500])
    actual = panel.loc[(panel["date"] == probe_date)
                       & (panel["ticker"] == ticker), "target"]
    trading_dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    probe = trading_dates[1000]
    cutoff = training_cutoff(trading_dates, probe)
    gap = trading_dates.searchsorted(probe) - trading_dates.searchsorted(cutoff)
    panel.to_csv(PANEL_FILE, index=False)
    print(f"\nSaved panel to {PANEL_FILE}")

if __name__ == "__main__":
    main()
        