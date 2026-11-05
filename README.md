# From market data to risk insight - case study

Three tasks in one project: an ECB exchange-rate ETL (1a), a daily FX-risk
dashboard (1b) and an index correlation analysis (2).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest        # tests, no network needed
ruff check .            # lint
```

Input files for Task 2 (`dim_master.csv`, `price_hist.parquet`) go into `data/`.

## Architecture

```
 ECB Data Portal API ──┐
 (SDMX REST, CSV)      │   src/ecb_client.py          src/etl.py                  data/risk.duckdb
                       ├─> SourceAdapter.fetch() ──>  checks -> upsert -> log ──> fx_rates
 Local CSV file  ──────┘   (EcbClient,                                            etl_load_log
 (offline / legacy)         CsvFileAdapter)                                           │
                                                                                      v
                                                                        src/fx_risk.py -> output/dashboard.html
```

All parameters are in `config.py`.

## Task 1a - ECB ETL

```bash
python -m src.etl --initial     # full history since 1999, table rebuilt
python -m src.etl --daily       # only new days
python -m src.etl --daily --source-file path/to/ecb.csv   # same flow from a file
```

**Source.** Series `EXR.D.USD.EUR.SP00.A`: daily ECB euro reference rate,
quoted as **USD per 1 EUR**. Requested as CSV from
`https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A?format=csvdata&startPeriod=...`.

**Design decisions**

| Topic | Decision | Why |
|-------|----------|-----|
| Storage | One local DuckDB file | No server, SQL access, fast, a single file to hand over |
| History | Full history from 1999 | The series is small (~7,000 rows); risk windows can be chosen later |
| Adapter | `SourceAdapter` with `fetch(start_date)` returning `(rate_date, rate)` | A new source only needs a new adapter; checks, load and log are reused |
| Initial mode | Delete + insert in one transaction | A clean rebuild; if anything fails, the old data stays |
| Daily mode | Request from last stored date minus 5 days, upsert on `(series_id, rate_date)` | Idempotent: running twice gives the same table; also picks up ECB corrections |
| Hard checks (stop the load) | valid dates, valid values, rate > 0, no duplicate dates | Bad data must never reach the risk numbers |
| Soft checks (warnings) | missing weekdays that are not TARGET holidays, weekend dates, daily moves > 5% | Worth a look, but no reason to block the dashboard |
| Errors | 30 s timeout, 3 attempts with backoff (2 s, 4 s) for timeouts, 5xx, 429; clear `FAILED` message and exit code 1 | A scheduler can alert on the exit code |
| Log | One row per run in `etl_load_log` | Full audit trail: when, which mode, how many new rows, which date range, check results |

**Gaps are normal.** The ECB publishes only on TARGET business days. No rate on
weekends, 1 January, Good Friday, Easter Monday, 1 May, 25 and 26 December. The gap
check knows these days and reports only other missing weekdays (for example early
years with different closing days).

**HTTP 404 = no data.** The API answers 404 "No results found" if the requested
period has no observation yet (e.g. a daily run before publication). The ETL logs
this as `NO_NEW_DATA`, not as an error.

**Inspect the data**

```bash
python -c "import duckdb; c=duckdb.connect('data/risk.duckdb'); print(c.sql('SELECT * FROM etl_load_log ORDER BY run_at DESC LIMIT 5')); print(c.sql('SELECT * FROM fx_rates ORDER BY rate_date DESC LIMIT 5'))"
```

**Automating the daily run.** The ECB publishes the reference rates at around
16:00 CET. The daily job runs at 16:30 CET on weekdays, with a second run at 18:00
as a safety net (safe because the load is idempotent).

Linux / macOS (cron):
```
CRON_TZ=Europe/Berlin
30 16 * * 1-5  cd /path/to/project && .venv/bin/python run_daily.py >> logs/daily.log 2>&1
0  18 * * 1-5  cd /path/to/project && .venv/bin/python run_daily.py >> logs/daily.log 2>&1
```

Windows (Task Scheduler):
```
schtasks /Create /TN "FX risk daily" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 16:30 ^
  /TR "C:\path\to\project\.venv\Scripts\python.exe C:\path\to\project\run_daily.py"
```

In a bank setup the same command would run in the central scheduler
(e.g. Airflow, Control-M) with alerting on exit code 1.

## Task 1b - FX-risk dashboard

```bash
python run_daily.py          # ETL daily (initial if the DB is empty) + rebuild dashboard
```

Open `output/dashboard.html` in any browser. It is one self-contained file
(Plotly embedded, works offline) and is rebuilt on every run. Building the dashboard
only: `python -m src.dashboard`.

**Position and quote direction.** The bank holds USD 100,000 and converts within
5 days. The ECB quotes **USD per 1 EUR** (e.g. 1.08), so

```
EUR value = 100,000 / rate
rate up   (stronger euro) -> fewer EUR -> loss
rate down (weaker euro)   -> more EUR  -> gain
```

With a 5-day log return `r = ln(S_t+5 / S_t)` the relative loss is `1 - exp(-r)`.

**Risk measures**

| | Choice | Why |
|---|---|---|
| Main measure | 5-day VaR 99%, historical simulation | Standard in banks, easy to explain, no distribution assumption |
| Additional | Expected Shortfall 97.5% | Basel FRTB measure; says how bad the tail is beyond VaR |
| Benchmark | Parametric VaR: normal distribution, mean 0, √5 scaling | Shows the effect of fat tails |
| Horizon | 5 trading days = 5 ECB publication days | Conversion "within 5 days"; we take the worst case: held for the full 5 days |
| Returns | Overlapping 5-day returns, no √5 scaling | Uses the real 5-day distribution (incl. autocorrelation and fat tails); √5 assumes i.i.d. returns. √5 version shown for comparison |
| Window | 750 days (~3 years) main, 250 days (~1 year) sensitivity | With 250 overlapping 5-day returns only ~50 are independent, the 1% quantile rests on 2-3 points. 3 years is more stable, 1 year reacts faster |
| Quantile | `numpy.quantile`, linear interpolation | Documented, reproducible |

**Backtest.** For every day t the VaR uses only data up to t (rolling window, no
look-ahead). It is compared with the loss realised over the next 5 days.
An exception is a realised loss above VaR. Reported:
exceptions over the last 250 days, Kupiec POF test, Basel traffic light
(0-4 green, 5-9 yellow, 10+ red), and the full history.

Limits of the backtest:
- Overlapping 5-day windows share 4 of 5 days, so exceptions are not independent and
  come in clusters. The Kupiec test assumes independence and is therefore too optimistic.
  A second Kupiec test on non-overlapping windows (every 5th day) is shown as a check.
- The Basel traffic light is defined for 1-day VaR on 250 days; here it is a guide only.

**Traceability.** Every number on the dashboard comes from `fx_rates` and is stored
again in the DuckDB table `fx_risk_history` (VaR, ES, benchmark, realised loss,
exception flag per day).

**Assumptions and limitations** (also on the dashboard): conversion at the ECB
reference rate without spread or fees; history represents the next 5 days; no
intraday moves (ECB rate is a 14:10 CET snapshot); VaR does not show the size of
losses beyond it.

## Task 2 - Index analysis

```bash
python -m src.index_analysis   # needs data/dim_master.csv and data/price_hist.parquet
```

Writes `output/index_analysis.html` (report with all charts, linked from the dashboard)
and CSV tables in `output/` (correlation matrices, sensitivity, flagged outliers).

**Question.** Are the indices more correlated in market distress?

**Method**
- Crisis day = VIX close above 30. Fixed, widely used stress level that does not move
  with the sample (about 10% of days). Sensitivity: VIX 20, 25, 35, 40 and the 80% quantile (24.4).
- VIX is only the crisis signal, not part of the correlations (30 equity indices remain).
- Log returns per index on its own trading days. Pearson correlation, pairwise complete.
- Average pairwise correlation = mean of the upper triangle of the matrix.
- Weekly returns (Friday to Friday) as a check against the time-zone effect.
- Forbes-Rigobon adjustment: `rho* = rho / sqrt(1 + delta * (1 - rho^2))`,
  `delta` = variance ratio crisis/normal of the S&P 500 minus 1.

**Result** (all numbers are reproduced by the script)

| Average pairwise correlation | normal | crisis | crisis, FR-adjusted |
|---|---|---|---|
| Daily, outliers excluded (main) | 0.48 | 0.60 | 0.27 |
| Weekly | 0.60 | 0.77 | 0.43 |

- Yes: correlations are clearly higher in distress, for every VIX threshold.
- After the volatility adjustment the effect disappears: the rise is explained by higher
  volatility (interdependence, not contagion). For risk the raw number matters:
  diversification is weakest when losses are largest.
- Asia is least connected on normal days and rises most in crisis; Europe is most connected.

**Data traps**

| Trap | Finding | Treatment |
|---|---|---|
| Duplicate key | IXIC and NDX share `US6311011026`; naive join 152,559 instead of 143,211 rows, two prices per day | Split by price level (Composite higher until 2020, NASDAQ 100 higher from mid-2021). 344 days in 2020-21 where both are within 1% and cross are set to missing. Checked against 3 published closes |
| Time zones | Asia, Europe, US close at different times | Weekly returns as comparison; weekly correlations are higher |
| Outliers | 26 flagged, e.g. SDAX +17% and OMX Helsinki +10% on 2013-03-05 (after a 3-month data gap), KOSPI +16.5% on 2026-07-31 | Rule: \|z\| > 5 vs. trailing 250-day std while the median other index has \|z\| < 1.5 (same or next day), or a return across a gap > 14 days. Flagged, listed in `output/index_outliers.csv`, excluded from the main result; result with outliers shown too |
| Weekend rows | 58 (special sessions India/Taiwan) | Dropped; the next close-to-close return covers the move |
| Volume 0 | 25% of rows | Volume not used |
| Start dates / holidays | UK 100 from 2010, TOPIX from 2011; different holidays | Own trading days per index, pairwise complete correlations, no forward-fill |

**Limitations.** Crisis correlation is partly a volatility effect (Forbes-Rigobon). The
adjustment assumes one source market and no common shock, so it over-corrects in global
crises: the truth lies between raw and adjusted. The VIX is US-centric. Crisis days cluster
in few episodes (2008, 2020). Local-currency index levels, no FX.
