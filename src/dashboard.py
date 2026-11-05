"""Build the board dashboard: one static HTML file (output/dashboard.html).

    python -m src.dashboard

Reads the rates and the ETL log from DuckDB, computes the risk figures with
src/fx_risk.py and writes a self-contained HTML page (Plotly embedded, works offline).
"""

from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

import config
from src import fx_risk

# Colours (dark theme). Series colours from a CVD-checked palette; status colours separate.
BLUE, ORANGE, GREY = "#3987e5", "#d95926", "#8a94a6"
GOOD, WARN, BAD = "#22c55e", "#eab308", "#ef4444"
TEXT, MUTED, GRID = "#e6eaf2", "#9aa4b5", "#1f2a3c"


# --- Data collection ---------------------------------------------------------


def last_etl_run(db_path: Path) -> dict:
    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute("SELECT * FROM etl_load_log ORDER BY run_at DESC LIMIT 1").df()
    con.close()
    return df.iloc[0].to_dict() if len(df) else {}


def collect(db_path: Path = config.DUCKDB_PATH) -> dict:
    """Everything the page shows, in one dict (easy to inspect and test)."""
    rates = fx_risk.load_rates(db_path)
    roll = fx_risk.rolling_risk(rates)
    fx_risk.save_results(roll, db_path)
    return {
        "rates": rates,
        "roll": roll,
        "snap": fx_risk.risk_snapshot(rates),
        "snap_short": fx_risk.risk_snapshot(rates, window=config.WINDOW_DAYS_SHORT),
        "bt": fx_risk.backtest_summary(roll),
        "etl": last_etl_run(db_path),
    }


# --- Small formatting helpers ------------------------------------------------


def eur(x: float, decimals: int = 0) -> str:
    return f"€{x:,.{decimals}f}"


def pct(x: float, decimals: int = 2) -> str:
    return f"{x * 100:.{decimals}f}%"


def sparkline(values: pd.Series, color: str = GOOD) -> str:
    """Tiny inline SVG line for a KPI card."""
    v = values.to_numpy()
    lo, hi = v.min(), v.max()
    span = (hi - lo) or 1
    pts = " ".join(f"{i * 100 / (len(v) - 1):.1f},{38 - (x - lo) / span * 34:.1f}"
                   for i, x in enumerate(v))
    return (f'<svg class="spark" viewBox="0 0 100 40" preserveAspectRatio="none">'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" '
            f'vector-effect="non-scaling-stroke"/></svg>')


def coverage_ring(share: float, color: str) -> str:
    """Donut showing the backtest coverage (share of days without exception)."""
    circ = 2 * 3.14159 * 42
    return (f'<svg viewBox="0 0 100 100" class="ring">'
            f'<circle cx="50" cy="50" r="42" stroke="{GRID}" stroke-width="9" fill="none"/>'
            f'<circle cx="50" cy="50" r="42" stroke="{color}" stroke-width="9" fill="none" '
            f'stroke-linecap="round" stroke-dasharray="{circ * share:.1f} {circ:.1f}" '
            f'transform="rotate(-90 50 50)"/>'
            f'<text x="50" y="50" class="ring-num">{share * 100:.1f}%</text>'
            f'<text x="50" y="66" class="ring-lbl">coverage</text></svg>')


# --- Charts ------------------------------------------------------------------


RANGE_BUTTONS = [dict(count=1, label="1M", step="month", stepmode="backward"),
                 dict(count=3, label="3M", step="month", stepmode="backward"),
                 dict(count=6, label="6M", step="month", stepmode="backward"),
                 dict(count=1, label="1Y", step="year", stepmode="backward"),
                 dict(count=3, label="3Y", step="year", stepmode="backward"),
                 dict(step="all", label="ALL")]


def style(fig: go.Figure, start, end, y_title: str, money: bool = False) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color=MUTED, size=12),
        margin=dict(l=8, r=8, t=8, b=8), height=300, hovermode="x unified",
        legend=dict(orientation="h", y=1.12, x=1, xanchor="right", font=dict(color=TEXT)),
        hoverlabel=dict(bgcolor="#0f1726", bordercolor=GRID, font=dict(color=TEXT)),
    )
    fig.update_xaxes(range=[start, end], gridcolor=GRID, showgrid=False, linecolor=GRID,
                     rangeselector=dict(buttons=RANGE_BUTTONS, bgcolor="#162033",
                                        activecolor="#25324a", font=dict(color=TEXT),
                                        x=0, y=1.12))
    fig.update_yaxes(title=y_title, gridcolor=GRID, zeroline=False,
                     tickprefix="€" if money else "", tickformat=",.0f" if money else ".3f")
    return fig


def fit_y(fig: go.Figure, start, pad: float = 0.08) -> go.Figure:
    """Set the y-range to the data visible in the initial x-window (Plotly would use all data)."""
    ys = [pd.Series(t.y, index=pd.to_datetime(t.x)) for t in fig.data]
    vis = pd.concat([y[y.index >= start] for y in ys]).dropna()
    lo, hi = vis.min(), vis.max()
    margin = (hi - lo) * pad or abs(hi) * pad
    return fig.update_yaxes(range=[lo - margin, hi + margin])


def rate_chart(rates: pd.Series) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=rates.index, y=rates.values, name="USD per EUR", mode="lines",
        line=dict(color=BLUE, width=2), fill="tozeroy", fillcolor="rgba(57,135,229,0.10)",
        hovertemplate="%{y:.4f} USD per EUR<extra></extra>"))
    end = rates.index[-1]
    start = end - pd.DateOffset(years=1)
    fig = fit_y(style(fig, start, end, "USD per 1 EUR"), start)
    fig.update_layout(showlegend=False)
    return fig


def risk_chart(roll: pd.DataFrame) -> go.Figure:
    var_eur = roll["var"] * roll["value_eur"]
    es_eur = roll["es"] * roll["value_eur"]
    par_eur = roll["param_var"] * roll["value_eur"]
    fig = go.Figure([
        go.Scatter(x=roll.index, y=var_eur, name="VaR 99% (historical)",
                   line=dict(color=BLUE, width=2), hovertemplate="€%{y:,.0f}"),
        go.Scatter(x=roll.index, y=es_eur, name="ES 97.5%",
                   line=dict(color=ORANGE, width=2), hovertemplate="€%{y:,.0f}"),
        go.Scatter(x=roll.index, y=par_eur, name="VaR 99% (normal benchmark)",
                   line=dict(color=GREY, width=1.5, dash="dot"), hovertemplate="€%{y:,.0f}"),
    ])
    end = roll.index[-1]
    start = end - pd.DateOffset(years=3)
    return fit_y(style(fig, start, end, "5-day risk in EUR", money=True), start)


def backtest_chart(roll: pd.DataFrame) -> go.Figure:
    done = roll.dropna(subset=["fwd_loss"])
    loss_eur = done["fwd_loss"] * done["value_eur"]
    var_eur = done["var"] * done["value_eur"]
    exc = done["exception"]
    fig = go.Figure([
        go.Scatter(x=done.index, y=loss_eur, name="Realised 5-day loss (+) / gain (-)",
                   line=dict(color=GREY, width=1), hovertemplate="€%{y:,.0f}"),
        go.Scatter(x=done.index, y=var_eur, name="VaR 99% forecast",
                   line=dict(color=BLUE, width=2), hovertemplate="€%{y:,.0f}"),
        go.Scatter(x=done.index[exc], y=loss_eur[exc], name="Exception (loss > VaR)",
                   mode="markers", marker=dict(color=BAD, size=8, line=dict(color="#0b1220",
                                                                            width=2)),
                   hovertemplate="€%{y:,.0f}"),
    ])
    end = done.index[-1]
    start = end - pd.DateOffset(years=3)
    return fit_y(style(fig, start, end, "EUR", money=True), start)


def fig_html(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"displayModeBar": False, "responsive": True})


# --- Page --------------------------------------------------------------------


def build_html(d: dict) -> str:
    s, ss, bt = d["snap"], d["snap_short"], d["bt"]
    etl, rates, roll = d["etl"], d["rates"], d["roll"]
    as_of = s["as_of"]
    h = config.HORIZON_DAYS

    # KPI card 1: position value and change over the last 5 trading days
    change = s["value_eur"] / s["value_eur_prev"] - 1
    up = change >= 0
    arrow, chg_cls = ("▲", "good") if up else ("▼", "bad")
    spark = sparkline(fx_risk.eur_value(rates.iloc[-60:]), GOOD if up else BAD)

    # KPI card 4: freshness of the data
    age = (pd.Timestamp(datetime.now(UTC).date()) - as_of).days
    fresh = age <= config.STALE_AFTER_DAYS
    etl_status = str(etl.get("status", "n/a"))
    etl_ok = etl_status in ("OK", "OK_WITH_WARNINGS", "NO_NEW_DATA")
    badge = ('<span class="badge good">✓ Up to date</span>' if fresh and etl_ok
             else f'<span class="badge bad">! Check data ({age} day{"s" if age != 1 else ""} old, '
                f'ETL {etl_status})</span>')

    # Backtest card
    zone = bt["basel_zone"]
    zone_color = {"Green": GOOD, "Yellow": WARN, "Red": BAD}[zone]
    coverage = 1 - bt["exceptions_last"] / bt["n_last"]
    # Kupiec is two-sided: it rejects too many AND too few exceptions.
    expected_exc = bt["n_last"] * (1 - config.VAR_CONFIDENCE)
    if bt["kupiec_p_last"] > 0.05:
        kupiec = '<span class="good">✓ Pass</span>'
    elif bt["exceptions_last"] < expected_exc:
        kupiec = '<span class="warn">Conservative</span>'  # fewer losses than VaR predicts
    else:
        kupiec = '<span class="bad">✗ Reject</span>'
    kupiec_note = (f"Expected {expected_exc:.1f} exceptions, p = {bt['kupiec_p_last']:.2f}")

    # Key observations (generated from the numbers, no hand-written claims)
    diff_param = s["var_eur"] / s["param_var_eur"] - 1
    rate_move = rates.iloc[-1] / rates.iloc[-1 - h] - 1
    obs = [
        f"Historical VaR is {abs(diff_param):.0%} {'above' if diff_param > 0 else 'below'} "
        f"the normal-distribution benchmark ({eur(s['param_var_eur'])})"
        + (" - the data have fatter tails than a normal distribution." if diff_param > 0 else "."),
        f"With a 1-year window the VaR would be {eur(ss['var_eur'])} "
        f"({'higher' if ss['var_eur'] > s['var_eur'] else 'lower'}: recent markets are "
        f"{'more' if ss['var_eur'] > s['var_eur'] else 'less'} volatile than the 3-year average).",
        f"EUR/USD moved {rate_move:+.2%} over the last {h} trading days "
        f"({'stronger' if rate_move > 0 else 'weaker'} euro).",
        f"Last data load: {etl_status} ({html.escape(str(etl.get('message', '')))[:90]}).",
    ]
    obs_html = "".join(f"<li>{o}</li>" for o in obs)

    # Method comparison table
    rows = [
        ("Historical simulation, 3-year window (main)", s["var_eur"], s["var_pct"]),
        ("Historical simulation, 1-year window", ss["var_eur"], ss["var_pct"]),
        ("Normal distribution, √5 scaling (benchmark)", s["param_var_eur"], s["param_var_pct"]),
        ("Historical 1-day VaR × √5", s["sqrt5_var_pct"] * s["value_eur"], s["sqrt5_var_pct"]),
        ("Expected Shortfall 97.5%, 3-year window", s["es_eur"], s["es_pct"]),
    ]
    table = "".join(f"<tr><td>{n}</td><td class='num'>{eur(v)}</td>"
                    f"<td class='num'>{pct(p)}</td></tr>" for n, v, p in rows)

    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    template = (Path(__file__).parent / "dashboard_template.html").read_text(encoding="utf-8")
    values = {
        "TITLE": html.escape(config.DASHBOARD_TITLE),
        "ORG": html.escape(config.DASHBOARD_ORG),
        "USD": f"{config.POSITION_USD:,.0f}",
        "H": str(h),
        "AS_OF": f"{as_of:%b %d, %Y}",
        "VALUE": eur(s["value_eur"], 2),
        "RATE": f"{s['rate']:.4f}",
        "CHG": f"{arrow} {change:+.2%}", "CHG_CLS": chg_cls, "SPARK": spark,
        "VAR": eur(s["var_eur"], 2), "VAR_PCT": pct(s["var_pct"]),
        "ES": eur(s["es_eur"], 2), "ES_PCT": pct(s["es_pct"]),
        "BADGE": badge,
        "DOT_COLOR": GOOD if fresh and etl_ok else BAD,
        "SENTENCE": (f"With 99% confidence, we do not expect to lose more than "
                     f"<b>{eur(s['var_eur'])}</b> on the USD {config.POSITION_USD:,.0f} position "
                     f"over the next {h} trading days. In the worst 2.5% of cases the average "
                     f"loss is <b>{eur(s['es_eur'])}</b>."),
        "RATE_CHART": fig_html(rate_chart(rates)),
        "RISK_CHART": fig_html(risk_chart(roll)),
        "BT_CHART": fig_html(backtest_chart(roll)),
        "RING": coverage_ring(coverage, zone_color),
        "EXC_LAST": f"{bt['exceptions_last']} / {bt['n_last']}",
        "KUPIEC": kupiec, "KUPIEC_NOTE": kupiec_note, "ZONE": zone, "ZONE_COLOR": zone_color,
        "BT_FULL": (f"Full history {bt['start']:%Y}-{bt['end']:%Y}: {bt['exceptions']} exceptions "
                    f"in {bt['n']:,} days ({pct(bt['rate'], 1)}, expected 1.0%), Kupiec p = "
                    f"{bt['kupiec_p']:.2f}. Non-overlapping windows only: "
                    f"{bt['exceptions_nonoverlap']} / {bt['n_nonoverlap']:,}, p = "
                    f"{bt['kupiec_p_nonoverlap']:.2f}."),
        "OBS": obs_html,
        "TABLE": table,
        "SCEN": f"{s['n_scenarios']:,}",
        "SERIES": config.SERIES_ID,
        "SOURCE": html.escape(str(etl.get("source", "n/a"))),
        "GENERATED": generated,
        "DATA_JSON": json.dumps({"as_of": f"{as_of:%Y-%m-%d}", "var_eur": round(s["var_eur"], 2),
                                 "es_eur": round(s["es_eur"], 2)}),
    }
    for key, val in values.items():
        template = template.replace("{{" + key + "}}", val)
    # Plotly's JavaScript last, so the placeholders above are never searched inside it.
    return template.replace("{{PLOTLY_JS}}", get_plotlyjs())


def build(db_path: Path = config.DUCKDB_PATH, out: Path = config.DASHBOARD_PATH) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(collect(db_path)), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(f"Dashboard written to {build()}")
