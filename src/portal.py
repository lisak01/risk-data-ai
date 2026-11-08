"""The portal: one web page that shows everything, built by `python run_all.py`.

Tabs (sidebar):
    Dashboard        board view: status bar, KPI cards, charts, one plain sentence
    Risk Analytics   method comparison, backtest chart, assumptions and limits
    Index Analysis   Task 2 report
    Data Pipeline    sources and cloud adapters, architecture, load log, database tables
    Quality          test results (pytest) and code check (ruff)
    Design Decisions why each choice was made, and where each case requirement is met

The page is one self-contained HTML file (Plotly embedded): no server, works offline,
can be sent to the board as a file.
"""

from __future__ import annotations

import base64
import html
import json
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import duckdb
from plotly.offline import get_plotlyjs

import config
from src import adapters, dashboard

TEMPLATE = Path(__file__).parent / "portal_template.html"
TEST_RESULTS = config.OUTPUT_DIR / "test_results.xml"  # written by run_all.py (pytest)
RUFF_RESULTS = config.OUTPUT_DIR / "ruff.json"  # written by run_all.py (ruff)

TEST_GROUPS = {"test_etl": "Task 1a - ETL and source adapters",
               "test_fx_risk": "Task 1b - FX risk and dashboard",
               "test_index_analysis": "Task 2 - index analysis"}


def esc(x) -> str:
    return html.escape(str(x))


# --- Data for the tabs -------------------------------------------------------


def pipeline_data(db_path: Path) -> dict:
    """Load log and table sizes from DuckDB (Data Pipeline tab)."""
    con = duckdb.connect(str(db_path), read_only=True)
    log = con.execute("""
        SELECT run_at, mode, source, rows_fetched, rows_new, min_date, max_date, status, message
        FROM etl_load_log ORDER BY run_at DESC LIMIT 10""").df()
    tables = {}
    for name in ("fx_rates", "etl_load_log", "fx_risk_history"):
        try:
            tables[name] = con.execute(f"SELECT count(*) FROM {name}").fetchall()[0][0]
        except duckdb.Error:
            tables[name] = None
    span = con.execute("SELECT min(rate_date), max(rate_date) FROM fx_rates").fetchall()[0]
    con.close()
    return {"log": log, "tables": tables, "span": span}


def test_results(path: Path = TEST_RESULTS) -> dict | None:
    """Read the pytest JUnit XML file. None if the tests were not run yet."""
    if not path.exists():
        return None
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    cases = []
    for tc in suite.iter("testcase"):
        if tc.find("failure") is not None or tc.find("error") is not None:
            status = "failed"
        elif tc.find("skipped") is not None:
            status = "skipped"
        else:
            status = "passed"
        cases.append({"module": tc.get("classname", "").split(".")[-1], "name": tc.get("name"),
                      "time": float(tc.get("time", 0)), "status": status})
    count = {k: sum(c["status"] == k for c in cases) for k in ("passed", "failed", "skipped")}
    return {"cases": cases, "total": len(cases), **count,
            "time": float(suite.get("time", 0)), "timestamp": suite.get("timestamp", "")[:16]}


def ruff_results(path: Path = RUFF_RESULTS) -> dict | None:
    if not path.exists():
        return None
    issues = json.loads(path.read_text(encoding="utf-8") or "[]")
    run_at = datetime.fromtimestamp(path.stat().st_mtime, UTC).strftime("%Y-%m-%d %H:%M")
    return {"issues": issues, "count": len(issues), "run_at": run_at}


# --- HTML pieces -------------------------------------------------------------


def status_item(ok: bool | None, label: str, value: str) -> str:
    """One field of the status bar. ok=None means 'not available' (grey)."""
    cls, icon = {True: ("good", "✓"), False: ("bad", "!"), None: ("muted", "–")}[ok]
    return (f'<div class="stat"><span class="stat-icon {cls}">{icon}</span>'
            f'<div><div class="label">{esc(label)}</div><div class="stat-val">{esc(value)}</div>'
            f'</div></div>')


def status_bar(fx: dict, tests: dict | None, ruff: dict | None) -> str:
    items = [
        status_item(fx["_fresh"], "Market data", f"Up to date ({fx['AS_OF']})" if fx["_fresh"]
                    else f"{fx['_age']} days old"),
        status_item(fx["_etl_ok"], "Last data load", fx["_etl_status"]),
        status_item(None if tests is None else tests["failed"] == 0, "Automated tests",
                    "not run yet" if tests is None
                    else f"{tests['passed']}/{tests['total'] - tests['skipped']} passed"),
        status_item(None if ruff is None else ruff["count"] == 0, "Code check (ruff)",
                    "not run yet" if ruff is None else f"{ruff['count']} issues"),
        status_item(fx["_zone"] == "Green", "VaR backtest", f"Basel {fx['_zone']}"),
    ]
    return "".join(items)


def table(headers: list[str], rows: list[list], num_cols: tuple = ()) -> str:
    head = "".join(f"<th{' class=num' if i in num_cols else ''}>{esc(h)}</th>"
                   for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f"<td{' class=num' if i in num_cols else ''}>{c}</td>"
                                    for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def pill(text: str, kind: str) -> str:
    return f'<span class="pill {kind}">{esc(text)}</span>'


def sources_html() -> str:
    rows = []
    for a in adapters.catalogue():
        driver = esc(a["driver"]) + ("" if a["driver_installed"] else " (not installed)")
        live = a["example"] == config.FX_SOURCE
        rows.append([f"<b>{esc(a['name'])}</b>" + (" " + pill("in use", "good") if live else ""),
                     esc(a["kind"]), f"<code>{esc(a['example'])}</code>", driver,
                     esc(a["use"]), esc(a["status"])])
    return table(["Source", "Type", "Address in config", "Driver", "Typical use", "Status"], rows)


def load_log_html(log) -> str:
    kind = {"OK": "good", "OK_WITH_WARNINGS": "warn", "NO_NEW_DATA": "muted", "FAILED": "bad"}
    rows = [[f"{r.run_at:%Y-%m-%d %H:%M}", esc(r.mode), esc(r.source),
             "" if r.rows_fetched != r.rows_fetched else f"{int(r.rows_fetched):,}",
             "" if r.rows_new != r.rows_new else f"{int(r.rows_new):,}",
             f"{str(r.min_date)[:10]} – {str(r.max_date)[:10]}"
             if r.min_date == r.min_date and r.min_date else "",
             pill(r.status, kind.get(r.status, "muted")), esc(str(r.message)[:80])]
            for r in log.itertuples()]
    return table(["Run (UTC)", "Mode", "Source", "Fetched", "New", "Date range", "Status",
                  "Message"], rows, num_cols=(3, 4))


def tests_html(tests: dict | None) -> str:
    if tests is None:
        return '<p class="note">No test results yet. Run <code>python run_all.py</code>.</p>'
    out = []
    for module, title in TEST_GROUPS.items():
        cases = [c for c in tests["cases"] if c["module"] == module]
        if not cases:
            continue
        rows = [[pill(c["status"], {"passed": "good", "failed": "bad"}.get(c["status"], "muted")),
                 esc(c["name"].removeprefix("test_").replace("_", " ")),
                 f"{c['time'] * 1000:.0f} ms"] for c in cases]
        ok = sum(c["status"] == "passed" for c in cases)
        out.append(f'<div class="card"><h2>{esc(title)} <span class="note">({ok}/{len(cases)} '
                   f'passed)</span></h2>{table(["Result", "What is tested", "Time"], rows, (2,))}'
                   f'</div>')
    return "".join(out)


def ruff_html(ruff: dict | None) -> str:
    if ruff is None:
        return '<p class="note">Not run yet.</p>'
    if ruff["count"] == 0:
        return (f'<p>{pill("0 issues", "good")} All files pass the configured rules '
                f'(pycodestyle, pyflakes, import order, bugbear, pyupgrade). '
                f'Checked {ruff["run_at"]} UTC.</p>')
    rows = [[esc(i["filename"].split("/")[-1]), str(i["location"]["row"]), esc(i["code"]),
             esc(i["message"])] for i in ruff["issues"][:30]]
    return table(["File", "Line", "Rule", "Message"], rows)


def logo_html() -> str:
    """Logo file if present (config.LOGO_PATH), else the text wordmark."""
    path = Path(config.LOGO_PATH)
    if path.exists():
        mime = "image/svg+xml" if path.suffix == ".svg" else f"image/{path.suffix[1:]}"
        data = base64.b64encode(path.read_bytes()).decode()
        alt = esc(config.BRAND_NAME)
        return f'<img class="logo-img" src="data:{mime};base64,{data}" alt="{alt}">'
    return f'<span class="bar"></span>{esc(config.BRAND_NAME)}'


def tool_use_done() -> bool:
    path = config.PROJECT_ROOT / "TOOL_USE.md"
    return path.exists() and "To be completed" not in path.read_text(encoding="utf-8")


# --- Build -------------------------------------------------------------------


def build(db_path: Path = config.DUCKDB_PATH, out: Path = config.DASHBOARD_PATH,
          index_res: dict | None = None, include_index: bool = True) -> Path:
    """Write the portal. index_res: results of index_analysis.analyse(), computed if missing."""
    fx = dashboard.fx_values(dashboard.collect(db_path))
    pipe = pipeline_data(db_path)
    tests, ruff = test_results(), ruff_results()

    if index_res is None and include_index and config.PRICE_HIST_PATH.exists():
        from src import index_analysis
        index_res = index_analysis.analyse()
    if index_res is not None:
        from src import index_analysis
        index_body = index_analysis.report_body(index_res)
        d = index_res["daily"]
        index_kpi = f"{d['avg_normal']:.2f} → {d['avg_crisis']:.2f}"
    else:
        index_body = ('<div class="card"><p>Index data not found. Put dim_master.csv and '
                      'price_hist.parquet into data/ and run '
                      '<code>python run_all.py</code>.</p></div>')
        index_kpi = "n/a"

    t = pipe["tables"]
    values = {
        **{k: v for k, v in fx.items() if not k.startswith("_")},
        "BRAND": esc(config.BRAND_NAME), "UNIT": esc(config.BRAND_UNIT), "LOGO": logo_html(),
        "TITLE": esc(config.DASHBOARD_TITLE),
        "STATUS_BAR": status_bar(fx, tests, ruff),
        "INDEX_BODY": index_body, "INDEX_KPI": index_kpi,
        "SOURCES": sources_html(), "FX_SOURCE": esc(config.FX_SOURCE),
        "LOAD_LOG": load_log_html(pipe["log"]),
        "ROWS_RATES": f"{t['fx_rates']:,}" if t["fx_rates"] else "0",
        "ROWS_LOG": f"{t['etl_load_log']:,}" if t["etl_load_log"] else "0",
        "ROWS_RISK": f"{t['fx_risk_history']:,}" if t["fx_risk_history"] else "0",
        "SPAN": f"{pipe['span'][0]} to {pipe['span'][1]}",
        "TESTS": tests_html(tests), "RUFF": ruff_html(ruff),
        "TESTS_TOTAL": "–" if tests is None else str(tests["total"]),
        "TESTS_PASSED": "–" if tests is None else str(tests["passed"]),
        "TESTS_FAILED": "–" if tests is None else str(tests["failed"]),
        "TESTS_TIME": "–" if tests is None else f"{tests['time']:.1f} s",
        "TESTS_AT": "not run yet" if tests is None else f"{tests['timestamp'].replace('T', ' ')}",
        "RUFF_COUNT": "–" if ruff is None else str(ruff["count"]),
        "TOOL_USE": pill("done", "good") if tool_use_done() else pill("open: author to fill in",
                                                                         "warn"),
        "GENERATED": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    }
    page = TEMPLATE.read_text(encoding="utf-8")
    for key, val in values.items():
        page = page.replace("{{" + key + "}}", str(val))
    # Plotly's JavaScript last, so the placeholders above are never searched inside it.
    page = page.replace("{{PLOTLY_JS}}", get_plotlyjs())

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out


if __name__ == "__main__":
    print(f"Portal written to {build()}")
