from __future__ import annotations

import os
import re


def load_simple_dotenv(path: str) -> None:
    """Load KEY=VALUE entries without requiring python-dotenv.

    Existing exported variables take priority. Spaces around '=' and matching
    single or double quotes around values are accepted.
    """
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].lstrip()
            key, separator, raw_value = line.partition("=")
            key = key.strip()
            if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                continue
            value = raw_value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ.setdefault(key, value)


# Resolve everything relative to this file, never the terminal working directory.
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DOTENV_FILE = os.path.join(MODEL_DIR, ".env")
load_simple_dotenv(DOTENV_FILE)


# ---------------------------------------------------------------------------
# Date range
# ---------------------------------------------------------------------------

START_DATE = "2016-01-01"
END_DATE = "2026-01-01"


# ---------------------------------------------------------------------------
# Asset universe — these names are also the news-search names
# ---------------------------------------------------------------------------

ETFS = {
    # Broad European market
    "EXSA.DE": "iShares STOXX Europe 600 UCITS ETF",

    # STOXX Europe 600 sector ETFs (correct ticker-to-sector mappings)
    "EXV1.DE": "iShares STOXX Europe 600 Banks UCITS ETF",
    "EXV4.DE": "iShares STOXX Europe 600 Health Care UCITS ETF",
    "EXV3.DE": "iShares STOXX Europe 600 Technology UCITS ETF",
    "EXV2.DE": "iShares STOXX Europe 600 Telecommunications UCITS ETF",
    "EXV6.DE": "iShares STOXX Europe 600 Basic Resources UCITS ETF",
    "EXV7.DE": "iShares STOXX Europe 600 Chemicals UCITS ETF",
    "EXH7.DE": "iShares STOXX Europe 600 Personal Household Goods UCITS ETF",
    "EXH8.DE": "iShares STOXX Europe 600 Retail UCITS ETF",

    # Country ETFs
    "EXS1.DE": "iShares Core DAX UCITS ETF Germany",
    "CAC.PA": "Amundi CAC 40 UCITS ETF France",
    "EWI": "iShares MSCI Italy ETF",
    "EWP": "iShares MSCI Spain ETF",
    "EWN": "iShares MSCI Netherlands ETF",
    "EWL": "iShares MSCI Switzerland ETF",
}

EQUITIES = {
    # Germany
    "SAP.DE": "SAP SE",
    "SIE.DE": "Siemens AG",
    "BAS.DE": "BASF SE",
    "VOW3.DE": "Volkswagen AG",
    "DTE.DE": "Deutsche Telekom",
    "ALV.DE": "Allianz SE",
    "BMW.DE": "BMW AG",
    "MBG.DE": "Mercedes-Benz Group",
    "MUV2.DE": "Munich Re",

    # France
    "MC.PA": "LVMH",
    "TTE.PA": "TotalEnergies",
    "OR.PA": "L'Oreal",
    "SAN.PA": "Sanofi",
    "BNP.PA": "BNP Paribas",
    "AIR.PA": "Airbus",
    "SU.PA": "Schneider Electric",
    "CAP.PA": "Capgemini",
    "AI.PA": "Air Liquide",

    # Netherlands
    "ASML.AS": "ASML Holding",
    "PHIA.AS": "Philips",
    "INGA.AS": "ING Group",
    "HEIA.AS": "Heineken",
    "AD.AS": "Ahold Delhaize",
    "WKL.AS": "Wolters Kluwer",

    # Spain
    "SAN.MC": "Banco Santander",
    "IBE.MC": "Iberdrola",
    "ITX.MC": "Inditex",
    "TEF.MC": "Telefonica",
    "ENG.MC": "Enagas",
    "BBVA.MC": "Banco Bilbao Vizcaya Argentaria",

    # Switzerland
    "NESN.SW": "Nestle",
    "NOVN.SW": "Novartis",
    "UBSG.SW": "UBS Group",
    "RO.SW": "Roche Holding",
    "ABBN.SW": "ABB Ltd",
    "CFR.SW": "Richemont",
    "HOLN.SW": "Holcim",

    # Italy
    "ENI.MI": "Eni SpA",
    "ISP.MI": "Intesa Sanpaolo",
    "UCG.MI": "UniCredit",
    "ENEL.MI": "Enel",
    "STMMI.MI": "STMicroelectronics",
}

ALL_TICKERS = {**ETFS, **EQUITIES}


# ---------------------------------------------------------------------------
# Foreign-exchange conversion
# ---------------------------------------------------------------------------

NON_EUR_TICKERS = {
    "NESN.SW": "CHFEUR=X",
    "NOVN.SW": "CHFEUR=X",
    "UBSG.SW": "CHFEUR=X",
    "RO.SW": "CHFEUR=X",
    "ABBN.SW": "CHFEUR=X",
    "CFR.SW": "CHFEUR=X",
    "HOLN.SW": "CHFEUR=X",
    "EWI": "USDEUR=X",
    "EWP": "USDEUR=X",
    "EWN": "USDEUR=X",
    "EWL": "USDEUR=X",
}


# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(MODEL_DIR, "data")

RAW_PRICES_FILE = os.path.join(DATA_DIR, "raw_prices.csv")
RAW_VOLUME_FILE = os.path.join(DATA_DIR, "raw_volume.csv")
BAD_TICKERS_FILE = os.path.join(DATA_DIR, "bad_tickers.csv")
PATCHED_PRICES_FILE = os.path.join(DATA_DIR, "patched_prices.csv")
EUR_PRICES_FILE = os.path.join(DATA_DIR, "prices_eur_unfilled.csv")
CLEAN_PRICES_FILE = os.path.join(DATA_DIR, "clean_prices_eur.csv")
NEWS_FILE = os.path.join(DATA_DIR, "news_headlines.csv")
SENTIMENT_FILE = os.path.join(DATA_DIR, "daily_sentiment.csv")
FEATURES_FILE = os.path.join(DATA_DIR, "features.csv")
RISKFREE_RATE_FILE = os.path.join(DATA_DIR, "euribor_3m.csv")


# ---------------------------------------------------------------------------
# API keys — values come only from .env or an exported environment variable
# ---------------------------------------------------------------------------

ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "").strip()
MARKETAUX_API_TOKEN = os.environ.get("MARKETAUX_API_TOKEN", "").strip()


def check_ticker_configuration() -> bool:
    """Validate ticker mappings and return True when they are consistent."""
    print("\nChecking ticker configuration...")
    passed = True

    overlap = sorted(set(ETFS) & set(EQUITIES))
    if overlap:
        print(f"FAIL: ETF/equity ticker overlap: {overlap}")
        passed = False

    mismatched = sorted(set(NON_EUR_TICKERS) - set(ALL_TICKERS))
    if mismatched:
        print(f"FAIL: FX mappings absent from ALL_TICKERS: {mismatched}")
        passed = False

    empty_names = sorted(ticker for ticker, name in ALL_TICKERS.items() if not name.strip())
    if empty_names:
        print(f"FAIL: Empty asset names: {empty_names}")
        passed = False

    if passed:
        print("PASS: ticker and currency mappings are consistent")
    return passed


def check_file_paths() -> bool:
    """Create the data directory and display configured output paths."""
    os.makedirs(DATA_DIR, exist_ok=True)
    print("\nChecking paths...")
    print(f"Model directory: {MODEL_DIR}")
    print(f"Data directory:  {DATA_DIR}")
    print(f"News file:       {NEWS_FILE}")
    print("PASS: data directory exists")
    return True


def print_configuration_summary() -> None:
    print("\n" + "=" * 72)
    print("DATA CONFIGURATION SUMMARY")
    print("=" * 72)
    print(f"Date range:     {START_DATE} to {END_DATE}")
    print(f"Universe size: {len(ALL_TICKERS)}")
    print(f"ETFs:          {len(ETFS)}")
    print(f"Equities:      {len(EQUITIES)}")
    print(f"Non-EUR:       {len(NON_EUR_TICKERS)}")
    print(f"Alpha Vantage key loaded: {bool(ALPHA_VANTAGE_API_KEY)}")
    print(f"Marketaux key loaded:     {bool(MARKETAUX_API_TOKEN)}")


def main() -> int:
    print_configuration_summary()
    passed = check_ticker_configuration() and check_file_paths()
    print("\nConfiguration:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())