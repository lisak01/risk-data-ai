"""Central configuration: every parameter lives here, nothing is hard-coded elsewhere.

Change a value here (e.g. the position size or the confidence level) and re-run
`python run_daily.py` - all outputs are recomputed from it.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"  # local only, not committed (see .gitignore)
OUTPUT_DIR = PROJECT_ROOT / "output"
DUCKDB_PATH = DATA_DIR / "risk.duckdb"

# --- Task 1a: ECB source -----------------------------------------------------
# Series key in the ECB Data Portal: daily (D) reference rate, USD per 1 EUR.
#   EXR = exchange rates dataflow, D = daily, USD = currency, EUR = denominator,
#   SP00 = spot, A = average (the official ECB euro foreign exchange reference rate).
ECB_BASE_URL = "https://data-api.ecb.europa.eu/service/data"
ECB_FLOW = "EXR"
ECB_SERIES_KEY = "D.USD.EUR.SP00.A"
SERIES_ID = f"{ECB_FLOW}.{ECB_SERIES_KEY}"  # name used in the database

# Full history: the euro reference rate starts on 1999-01-04.
INITIAL_START_DATE = "1999-01-01"

# Daily mode re-requests a few days before the last stored date. The upsert makes
# this harmless and it picks up the rare case that the ECB corrects a recent value.
DAILY_LOOKBACK_DAYS = 5

# HTTP behaviour of the adapter.
HTTP_TIMEOUT_SECONDS = 30
HTTP_MAX_RETRIES = 3  # attempts in total
HTTP_BACKOFF_SECONDS = 2  # waits 2s, 4s, ... between attempts

# Data-quality thresholds.
MAX_ABS_DAILY_CHANGE = 0.05  # |rate change| > 5% in one day -> warning, not an error

# --- Task 1b: FX position and risk model ---------------------------------------
# The bank holds USD 100,000 and converts them into EUR within 5 days.
# ECB quote is USD per 1 EUR, so: EUR value = POSITION_USD / rate.
POSITION_USD = 100_000

HORIZON_DAYS = 5  # trading days (= ECB publication days)
VAR_CONFIDENCE = 0.99  # main measure: 5-day VaR 99%
ES_CONFIDENCE = 0.975  # additional measure: Expected Shortfall 97.5%

# Observation window for historical simulation, in trading days.
WINDOW_DAYS = 750  # main: ~3 years -> ~750 overlapping 5-day scenarios
WINDOW_DAYS_SHORT = 250  # sensitivity: ~1 year, reacts faster but is noisier

# Backtest: Basel traffic light uses the last 250 observations.
BACKTEST_DAYS = 250

# Data older than this many calendar days is flagged as stale on the dashboard.
STALE_AFTER_DAYS = 4

# Text in the dashboard header (no logos or personal data in the repo).
DASHBOARD_TITLE = "FX Risk Monitor"
DASHBOARD_ORG = "Financial Risk"
DASHBOARD_PATH = OUTPUT_DIR / "dashboard.html"

# --- Task 2: index analysis ----------------------------------------------------
DIM_MASTER_PATH = DATA_DIR / "dim_master.csv"
PRICE_HIST_PATH = DATA_DIR / "price_hist.parquet"
INDEX_OUTPUT_PATH = OUTPUT_DIR / "index_analysis.html"

VIX_TICKER = "VIX.INDX"  # crisis signal only, not part of the correlations
VIX_CRISIS_LEVEL = 30  # main definition: VIX close above 30 = market distress
VIX_SENSITIVITY_LEVELS = [20, 25, 30, 35, 40]  # plus the 80% quantile, see the analysis

# NASDAQ Composite and NASDAQ 100 share asset_key US6311011026 (two prices per day).
NASDAQ_KEY = "US6311011026"
NASDAQ_AMBIGUOUS_RATIO = 1.01  # closer than 1% -> identity cannot be told from the level
# Published closing values used to check the split (source: Nasdaq index history).
NASDAQ_ANCHORS = {  # date: (IXIC, NDX)
    "2008-01-02": (2609.63, 2049.71),
    "2021-12-31": (15644.97, 16320.08),
    "2024-12-31": (19310.79, 21012.17),
}

# Outliers: flag, do not delete.
OUTLIER_Z = 5  # |return| > 5 trailing standard deviations ...
OUTLIER_MARKET_Z = 1.5  # ... while the median other index moved less than 1.5 std
OUTLIER_VOL_WINDOW = 250  # trading days for the trailing standard deviation
GAP_DAYS = 14  # a return over more than 14 calendar days spans a data gap

ROLLING_CORR_DAYS = 63  # ~3 months for the rolling average correlation
