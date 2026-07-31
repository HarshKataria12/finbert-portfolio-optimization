"""
05_collect_riskfree_rate.py

Downloads the 3-month Euribor rate from the ECB's Statistical Data
Warehouse, confirmed in the Week 1-2 plan as the risk-free rate used in
Sharpe and Sortino ratio calculations later in the backtest (Week 9-10).

This is grouped with the Week 3-4 data-collection scripts for
completeness, since it's a "confirm data sources" item from Week 1-2 that
hadn't been implemented in code yet. It isn't used again until evaluation.

Run any time after collecting prices. Output: data/euribor_3m.csv

IMPORTANT: needs real internet access to the ECB's Statistical Data
Warehouse (data.ecb.europa.eu). The ECB does not offer a simple JSON
API for this specific series in the way Yahoo Finance does for prices,
so this uses their SDMX-based data service, which returns CSV directly
from a stable URL — no API key needed, but the exact URL format can
occasionally change if the ECB restructures their data portal.
"""

import os
import pandas as pd
import requests

from data_config import DATA_DIR, RISKFREE_RATE_FILE, START_DATE, END_DATE

# ECB Statistical Data Warehouse SDMX endpoint for the 3-month Euribor series.
# Series key FM.M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA is the standard monthly
# 3-month Euribor series. If this exact key stops working (the ECB does
# restructure these occasionally), search data.ecb.europa.eu for
# "EURIBOR 3-month" and copy the series key shown in its own API link.
ECB_SERIES_KEY = "FM.M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA"
ECB_BASE_URL = "https://data-api.ecb.europa.eu/service/data"


ECB_DATAFLOW    = "FM"                              # dataflow ID  ← split here
ECB_SERIES_KEY  = "M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA"  # actual series key
ECB_BASE_URL    = "https://data-api.ecb.europa.eu/service/data"


def fetch_euribor_3m(start_date=START_DATE, end_date=END_DATE):
    url = f"{ECB_BASE_URL}/{ECB_DATAFLOW}/{ECB_SERIES_KEY}"
    params = {
        "format":      "csvdata",
        "startPeriod": start_date[:7],   # YYYY-MM
        "endPeriod":   end_date[:7],
        "detail":      "dataonly",       # drops metadata columns → cleaner CSV
    }
    headers = {"Accept": "text/csv"}     # belt-and-suspenders with the format param
    print(f"Requesting Euribor data from {url} ...")
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()

    from io import StringIO
    df = pd.read_csv(StringIO(resp.text))

    if "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
        raise RuntimeError(
            "Unexpected ECB response format. Columns received: "
            + str(df.columns.tolist())
        )

    result = df[["TIME_PERIOD", "OBS_VALUE"]].copy()
    result.columns = ["date", "euribor_3m_rate"]
    result["date"] = pd.to_datetime(result["date"])
    result["euribor_3m_rate"] = pd.to_numeric(result["euribor_3m_rate"], errors="coerce")
    return result.sort_values("date").reset_index(drop=True)

def expand_to_daily(monthly_rates):
    """
    The Euribor series from the ECB is monthly; forward-fill it onto a
    daily calendar so it lines up with the daily price/feature data used
    everywhere else in the pipeline. This is standard practice for a
    slow-moving rate like this one, matching how the rest of the pipeline
    forward-fills price gaps for closed markets.
    """
    monthly_rates = monthly_rates.set_index("date")
    daily_index = pd.date_range(monthly_rates.index.min(), monthly_rates.index.max(), freq="D")
    daily = monthly_rates.reindex(daily_index).ffill()
    daily.index.name = "date"
    return daily.reset_index()


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    try:
        monthly = fetch_euribor_3m()
    except Exception as e:
        print(f"ERROR: could not fetch Euribor data: {str(e)[:300]}")
        print("This needs real internet access to data.ecb.europa.eu. "
              "Run this on your own machine or in Colab, not a locked-down sandbox.")
        return

    daily = expand_to_daily(monthly)
    daily.to_csv(RISKFREE_RATE_FILE, index=False)

    print(f"\nSaved {len(daily)} daily risk-free rate observations to {RISKFREE_RATE_FILE}")
    print(f"Date range: {daily['date'].min()} to {daily['date'].max()}")
    print(f"Latest rate: {daily['euribor_3m_rate'].iloc[-1]:.3f}%")


if __name__ == "__main__":
    main()