"""
Latest-date predictions per asset, using the corrected pipeline.

Requires the corrected data_config.py, forecast_model.py, and a fresh
run of finance_model.py / finbert_sentiment.py / build_features.py first
-- this script does not re-derive currency conversion or FinBERT scoring
itself, it reads features.csv and clean_prices_eur.csv as already
built, so it only reflects the fixes if those files were regenerated
under the corrected code.

- Model 3 (XGBoost) and Model 4 (XGBoost + FinBERT) use the exact same
  ReturnForecaster class, feature sets, embargo and training window as
  the actual thesis backtest -- just run once more, at the most recent
  possible rebalancing date, to see what they currently forecast.
- Model 1 (Markowitz) has no per-asset "prediction" of its own; it uses
  the trailing 3-year historical mean return as its input, shown here
  for honest comparison.
- Model 2 (HRP) is intentionally excluded -- it does not forecast
  returns at all.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import json

from forecast_model import ReturnForecaster, training_cutoff

TECHNICAL_FEATURES = ["mom_5d", "mom_10d", "mom_20d", "realized_vol_20d",
                       "ma_ratio_50_200", "rsi_14d", "relative_volume"]
SENTIMENT_FEATURES = ["sentiment_mean", "sentiment_std", "sentiment_max", "headline_count"]

features = pd.read_csv("data/features.csv", index_col=0, parse_dates=True)
prices = pd.read_csv("data/clean_prices_eur.csv", index_col=0, parse_dates=True)
log_prices = np.log(prices)

# Build the 21-day forward target exactly as the real pipeline does
target = (log_prices.shift(-21) - log_prices).stack()
target.index.names = ["date", "ticker"]
target.name = "target"

panel = features.reset_index().rename(columns={"index": "date"})
panel = panel.merge(target.reset_index(), on=["date", "ticker"], how="left")

trading_dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
latest_date = trading_dates[-1]

def latest_forecast(feature_cols, label):
    fc = ReturnForecaster(feature_cols)
    preds = fc.fit_predict(panel, trading_dates, latest_date, refit=True)
    if preds is None:
        return pd.Series(dtype=float, name=label)
    preds.name = label
    return preds

pred_xgb = latest_forecast(TECHNICAL_FEATURES, "xgb")
pred_xgb_sent = latest_forecast(TECHNICAL_FEATURES + SENTIMENT_FEATURES, "xgb_sentiment")

# Model 1 proxy: trailing 3-year (756-day) annualised historical mean return
window = log_prices.loc[:latest_date].tail(756)
daily_mean = window.diff().mean()
hist_mean_annualised = daily_mean * 252

results = pd.DataFrame({
    "xgb_1m_forecast": pred_xgb,
    "xgb_sentiment_1m_forecast": pred_xgb_sent,
    "markowitz_hist_mean_annualised": hist_mean_annualised,
}).dropna(subset=["xgb_1m_forecast"])

results = results.sort_values("xgb_sentiment_1m_forecast", ascending=False)
results.to_csv("data/results/latest_predictions.csv")
print(f"Latest rebalancing-style date used: {latest_date.date()}")
print(f"Assets with a valid forecast: {len(results)}")
print(results.head(10).to_string())
print("...")
print(results.tail(5).to_string())

# Save a compact JSON for the diagram
out = results.reset_index().rename(columns={"index": "ticker"})
out.to_json("data/results/latest_predictions.json", orient="records")