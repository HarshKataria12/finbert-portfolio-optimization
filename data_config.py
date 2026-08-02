"""
data_config.py

Central configuration for the thesis data pipeline.

Contains:
    - Historical date range
    - ETF and equity universe
    - Foreign-exchange conversion mappings
    - Input and output file paths
    - Alpha Vantage API-key configuration

The data directory is resolved relative to this file, so the scripts work
regardless of the terminal's current working directory.
"""

import os


# ---------------------------------------------------------------------------
# Date Range
# ---------------------------------------------------------------------------
# Extra history before the 2019 backtest start is needed so the first
# walk-forward training window has enough observations.

START_DATE = "2019-01-01"
END_DATE = "2025-12-31"


# ---------------------------------------------------------------------------
# ETF Universe
# ---------------------------------------------------------------------------

ETFS = {
    # Broad European market
    "EXSA.DE": "iShares STOXX Europe 600",

    # STOXX Europe 600 sector ETFs
    "EXV1.DE": "iShares STOXX Europe 600 Banks",
    "EXV4.DE": "iShares STOXX Europe 600 Health Care",
    "EXV3.DE": "iShares STOXX Europe 600 Technology",
    "EXV2.DE": "iShares STOXX Europe 600 Oil & Gas",
    "EXV6.DE": "iShares STOXX Europe 600 Industrial Goods",
    "EXV7.DE": "iShares STOXX Europe 600 Retail",
    "EXH7.DE": "iShares STOXX Europe 600 Food & Beverage",
    "EXH8.DE": "iShares STOXX Europe 600 Utilities",

    # Country ETFs
    "EXS1.DE": "iShares Core DAX (Germany)",
    "CAC.PA": "Amundi CAC 40 (France)",
    "EWI": "iShares MSCI Italy",
    "EWP": "iShares MSCI Spain",
    "EWN": "iShares MSCI Netherlands",
    "EWL": "iShares MSCI Switzerland",
}


# ---------------------------------------------------------------------------
# Equity Universe
# ---------------------------------------------------------------------------

EQUITIES = {
    # Germany
    "SAP.DE": "SAP",
    "SIE.DE": "Siemens",
    "BAS.DE": "BASF",
    "VOW3.DE": "Volkswagen",
    "DTE.DE": "Deutsche Telekom",
    "ALV.DE": "Allianz",
    "BMW.DE": "BMW",
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
    "ASML.AS": "ASML",
    "PHIA.AS": "Philips",
    "INGA.AS": "ING Group",
    "HEIA.AS": "Heineken",
    "AD.AS": "Ahold Delhaize",
    "WKL.AS": "Wolters Kluwer",

    # Spain
    "SAN.MC": "Banco Santander",
    "IBE.MC": "Iberdrola",
    "ITX.MC": "Inditex",
    "BBVA.MC": "BBVA",
    "TEF.MC": "Telefonica",
    "FER.MC": "Ferrovial",

    # Switzerland
    "NESN.SW": "Nestle",
    "NOVN.SW": "Novartis",
    "UBSG.SW": "UBS",

    # Roche voting shares are used instead of ROG.SW because RO.SW
    # downloads more reliably in the current pipeline.
    "RO.SW": "Roche Holding AG",

    "ABBN.SW": "ABB",
    "CFR.SW": "Richemont",
    "HOLN.SW": "Holcim",

    # Italy
    "ENI.MI": "ENI",
    "ISP.MI": "Intesa Sanpaolo",
    "UCG.MI": "UniCredit",
    "ENEL.MI": "Enel",
    "STMMI.MI": "STMicroelectronics",
}


# Combined asset universe

ALL_TICKERS = {
    **ETFS,
    **EQUITIES,
}


# ---------------------------------------------------------------------------
# Foreign-Exchange Conversion
# ---------------------------------------------------------------------------
# These assets are not already priced in EUR and require conversion.
#
# Every key must exactly match a ticker in ALL_TICKERS. A mismatch would
# cause the downstream conversion process to skip that asset.

NON_EUR_TICKERS = {
    "NESN.SW": "CHFEUR=X",
    "NOVN.SW": "CHFEUR=X",
    "UBSG.SW": "CHFEUR=X",
    "RO.SW": "CHFEUR=X",
}


# ---------------------------------------------------------------------------
# File Paths
# ---------------------------------------------------------------------------
# data_config.py is expected to be stored in:
#
#     /Users/harshkataria/Desktop/Thesis/model/data_config.py
#
# Data will therefore be stored in:
#
#     /Users/harshkataria/Desktop/Thesis/model/data
#
# Using absolute paths based on __file__ prevents errors when a script is
# launched from a different terminal directory.

MODEL_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_DIR = os.path.join(
    MODEL_DIR,
    "data"
)


# Price-pipeline files

RAW_PRICES_FILE = os.path.join(
    DATA_DIR,
    "raw_prices.csv"
)

RAW_VOLUME_FILE = os.path.join(
    DATA_DIR,
    "raw_volume.csv"
)

BAD_TICKERS_FILE = os.path.join(
    DATA_DIR,
    "bad_tickers.csv"
)

PATCHED_PRICES_FILE = os.path.join(
    DATA_DIR,
    "patched_prices.csv"
)

EUR_PRICES_FILE = os.path.join(
    DATA_DIR,
    "prices_eur_unfilled.csv"
)

CLEAN_PRICES_FILE = os.path.join(
    DATA_DIR,
    "clean_prices_eur.csv"
)


# News-pipeline input

NEWS_FILE = os.path.join(
    DATA_DIR,
    "news_headlines.csv"
)


# FinBERT sentiment output

SENTIMENT_FILE = os.path.join(
    DATA_DIR,
    "daily_sentiment.csv"
)
# Feature-engineering output
FEATURES_FILE = os.path.join(
    DATA_DIR,
    "features.csv"
)
RISKFREE_RATE_FILE = os.path.join(
    DATA_DIR,
    "euribor_3m.csv"
)

# ---------------------------------------------------------------------------
# Alpha Vantage
# ---------------------------------------------------------------------------
# The API key is read from an environment variable instead of being stored
# directly in this source file.
#
# Set it in the terminal with:
#
#     export ALPHA_VANTAGE_API_KEY="your_real_key_here"

ALPHA_VANTAGE_API_KEY = os.environ.get(
    "ALPHA_VANTAGE_API_KEY",
    "YOUR_KEY_HERE"
)


# ---------------------------------------------------------------------------
# Configuration Tests
# ---------------------------------------------------------------------------

def check_ticker_configuration():
    """Check that currency-conversion tickers exist in the universe."""
    print("\nChecking foreign-exchange ticker configuration...")

    mismatched_tickers = [
        ticker
        for ticker in NON_EUR_TICKERS
        if ticker not in ALL_TICKERS
    ]

    if mismatched_tickers:
        print(
            "WARNING: The following NON_EUR_TICKERS entries "
            "do not exist in ALL_TICKERS:"
        )

        for ticker in mismatched_tickers:
            print(f"  - {ticker}")

        print(
            "These tickers may be skipped during EUR conversion."
        )

        return False

    print(
        "PASS: Every NON_EUR_TICKERS entry exists "
        "in ALL_TICKERS."
    )

    return True


def check_file_paths():
    """Create the data directory and print all configured paths."""
    print("\nChecking file-path configuration...")

    os.makedirs(
        DATA_DIR,
        exist_ok=True
    )

    print(f"Model directory:  {MODEL_DIR}")
    print(f"Data directory:   {DATA_DIR}")
    print(f"News input:       {NEWS_FILE}")
    print(f"Sentiment output: {SENTIMENT_FILE}")
    print(f"Features output:  {FEATURES_FILE}")
    print(f"Risk-free rate output:  {RISKFREE_RATE_FILE}")

    if not os.path.isdir(DATA_DIR):
        print("FAIL: Data directory is unavailable.")
        return False

    print("PASS: Data directory exists.")

    if os.path.exists(NEWS_FILE):
        print("PASS: News-headline file exists.")
    else:
        print("WARNING: News-headline file does not exist.")
        print(f"Expected location: {NEWS_FILE}")
        print("Run the news-collection script first.")

    return True


def print_configuration_summary():
    """Print a summary of the configured research universe."""
    print("\n" + "=" * 72)
    print("DATA CONFIGURATION SUMMARY")
    print("=" * 72)

    print(f"\nStart date: {START_DATE}")
    print(f"End date:   {END_DATE}")

    print(
        f"\nUniverse size: {len(ALL_TICKERS)} assets"
    )

    print(
        f"ETFs:           {len(ETFS)}"
    )

    print(
        f"Equities:       {len(EQUITIES)}"
    )

    print(
        f"Non-EUR assets: {len(NON_EUR_TICKERS)}"
    )


def main():
    """Run configuration tests when this file is executed directly."""
    print_configuration_summary()

    ticker_test_passed = check_ticker_configuration()
    path_test_passed = check_file_paths()

    print("\n" + "=" * 72)
    print("CONFIGURATION TEST RESULTS")
    print("=" * 72)

    print(
        "Ticker configuration: "
        + ("PASS" if ticker_test_passed else "FAIL")
    )

    print(
        "File-path configuration: "
        + ("PASS" if path_test_passed else "FAIL")
    )

    if ticker_test_passed and path_test_passed:
        print(
            "\nThe configuration is ready for the "
            "data-processing scripts."
        )
    else:
        print(
            "\nCorrect the reported configuration issues "
            "before running the pipeline."
        )


if __name__ == "__main__":
    main()