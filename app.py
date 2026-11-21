"""GridPulse AI — Energy Demand Forecasting & Consumption Intelligence (Streamlit).

Run: streamlit run app.py
"""
from __future__ import annotations

import hashlib
import html
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from gridpulse.core import (FEATURE_INFO, MAX_HOURS, MIN_HOURS, TEMP_MODES, Tariff,
                            anomalies, anomaly_scores, demo, energy_cost, evaluate, evaluation_json,
                            forecast, monthly_bill, validate_report)

ASSETS = Path(__file__).parent / "assets"
WEBSITE = "https://zubairlabs.com"
REPO = "https://github.com/muhammadzubair-data/gridpulse-ai-energy-demand-intelligence"
CYAN, EMERALD, BLUE, AMBER, RED, SLATE = "#22D3EE", "#10B981", "#3B82F6", "#F59E0B", "#EF4444", "#94A3B8"
ACTUAL, FORECAST, MODEL = "#0EA5E9", "#10B981", "#A855F7"
MAX_UPLOAD_MB = 15

st.set_page_config(page_title="GridPulse AI | ZubairLabs", page_icon=str(ASSETS / "icon.svg"), layout="wide",
                   initial_sidebar_state="expanded",
                   menu_items={"About": "GridPulse AI by ZubairLabs — energy demand forecasting on synthetic demonstration data.",
                               "Report a bug": REPO + "/issues"})

st.markdown(f"""<style>
.block-container {{max-width: 1440px; padding-top: 3.2rem; padding-bottom: 3rem;}}
.gp-hero {{padding: 24px 28px; border-radius: 20px; margin-bottom: 16px; color: #F8FAFC;
  background: radial-gradient(900px 260px at 0% 0%, rgba(34,211,238,.26), transparent 60%),
              radial-gradient(700px 260px at 100% 100%, rgba(16,185,129,.28), transparent 60%),
              linear-gradient(120deg, #041026 0%, #071A36 55%, #08304A 100%);
  border: 1px solid rgba(148,163,184,.22); box-shadow: 0 12px 32px rgba(2,8,23,.22);}}
.gp-hero .eyebrow {{letter-spacing: .2em; font-size: 11px; font-weight: 700; color: {CYAN}; text-transform: uppercase;}}
.gp-hero h1 {{font-size: 32px; margin: 6px 0 4px 0; padding: 0; color: #F8FAFC; font-weight: 800;}}
.gp-hero p {{margin: 0; color: #CBD5E1;}}
.pill {{display: inline-block; margin: 10px 6px 0 0; padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 600;
  background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.18); color: #E2E8F0;}}
.pill.warn {{background: rgba(245,158,11,.18); border-color: rgba(245,158,11,.5); color: #FDE68A;}}
.pill.real {{background: rgba(16,185,129,.18); border-color: rgba(16,185,129,.5); color: #A7F3D0;}}
.kpi-grid {{display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 6px 0 14px 0;}}
.kpi {{border-radius: 16px; padding: 14px 16px; border: 1px solid rgba(148,163,184,.28);
  background: linear-gradient(160deg, rgba(34,211,238,.10), rgba(16,185,129,.06));}}
.kpi .label {{font-size: 11.5px; font-weight: 700; opacity: .7; text-transform: uppercase; letter-spacing: .06em;}}
.kpi .value {{font-size: 26px; font-weight: 800; margin-top: 4px; line-height: 1.1;}}
.kpi .sub {{font-size: 12px; opacity: .65; margin-top: 4px;}}
.sec-title {{font-size: 24px; font-weight: 800; margin: 2px 0;}}
.sec-sub {{opacity: .7; margin-bottom: 14px; font-size: 14px;}}
.note {{border-left: 4px solid {EMERALD}; padding: 10px 14px; border-radius: 10px; background: rgba(16,185,129,.09); margin: 6px 0 12px 0;}}
.legend-key {{display:inline-flex; align-items:center; gap:6px; margin-right:16px; font-size:13px;}}
.legend-key i {{display:inline-block; width:22px; height:0; border-top:3px solid;}}
.footer {{opacity: .6; font-size: 12px; text-align: center; margin-top: 28px;}}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def kpis(items):
    e = html.escape
    st.markdown("<div class='kpi-grid'>" + "".join(
        f"<div class='kpi'><div class='label'>{e(a)}</div><div class='value'>{e(b)}</div><div class='sub'>{e(c)}</div></div>"
        for a, b, c in items) + "</div>", unsafe_allow_html=True)


def section(title, sub):
    st.markdown(f"<div class='sec-title'>{html.escape(title)}</div><div class='sec-sub'>{html.escape(sub)}</div>",
                unsafe_allow_html=True)


def note(text):
    st.markdown(f"<div class='note'>{text}</div>", unsafe_allow_html=True)


def legend_keys(model=True):
    st.markdown(
        f"<span class='legend-key'><i style='border-color:{ACTUAL}'></i>Actual history ({html.escape(SOURCE_WORD)})</span>"
        + (f"<span class='legend-key'><i style='border-color:{MODEL}; border-top-style:dotted'></i>Model estimate on past hours</span>" if model else "")
        +
        f"<span class='legend-key'><i style='border-color:{FORECAST}; border-top-style:dashed'></i>Model forecast (future)</span>",
        unsafe_allow_html=True)


def chart(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="", title_font=dict(size=15),
                      hovermode="x unified")
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def csv_bytes(df: pd.DataFrame) -> bytes:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].map(lambda v: "'" + v if isinstance(v, str) and v[:1] in ("=", "+", "-", "@") else v)
    return out.to_csv(index=False).encode()


def rnd(df: pd.DataFrame, n: int) -> pd.DataFrame:
    num = df.select_dtypes("number").columns
    return df.assign(**{c: df[c].round(n) for c in num})


def fmt_money(x, cur):
    return f"{cur} {x:,.0f}" if abs(x) >= 100 else f"{cur} {x:,.2f}"


# ------------------------------------------------------------------ data + model (cached)
@st.cache_data(show_spinner=False, max_entries=4)
def load_demo(days: int, seed: int) -> pd.DataFrame:
    return demo(days, seed)


# Each evaluation holds two Random Forests plus backtest tables (about 30-70 MB), so keep only a few
# and let uploaded-data entries expire; widget changes elsewhere never retrain.
@st.cache_resource(show_spinner=False, max_entries=3, ttl=3600)
def get_eval(key: str, _d: pd.DataFrame):
    return evaluate(_d)


@st.cache_data(show_spinner=False, max_entries=8, ttl=3600)
def get_flags(key: str, rate: float, _d: pd.DataFrame) -> pd.DataFrame:
    return anomalies(_d, rate)


@st.cache_data(show_spinner=False, max_entries=24, ttl=3600)
def get_forecast(key: str, horizon: int, mode: str, offset: float, _d: pd.DataFrame, _model) -> pd.DataFrame:
    return forecast(_d, horizon, _model, mode, offset)


# ------------------------------------------------------------------ sidebar / data source
ss = st.session_state
st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "icon.svg"), link=WEBSITE, size="large")

with st.sidebar:
    st.markdown("**Data source**")
    has_upload = "upload_df" in ss
    options = ["Synthetic demo"] + ([f"Uploaded: {ss['upload_name']}"] if has_upload else [])
    source = st.radio("Data source", options, index=len(options) - 1 if ss.get("use_upload") and has_upload else 0,
                      label_visibility="collapsed", key="source_choice")
    use_upload = has_upload and source != "Synthetic demo"
    ss["use_upload"] = use_upload
    if not use_upload:
        days = st.select_slider("History length (days)", [60, 90, 120, 180, 270, 365, 540, 730], value=365,
                                help="Length of the synthetic hourly history.")
        seed = st.number_input("Seed", 0, 9999, 42, help="Random seed: same seed, same data and results.")
    else:
        st.caption("Upload a different file or switch back on **Data Explorer**.")
    if not has_upload:
        st.caption("Have your own meter data? Upload a CSV on **Data Explorer**.")

if use_upload:
    DATA = ss["upload_df"]
    KEY = "upload-" + ss["upload_hash"]
    SOURCE_WORD = "uploaded"
    raw_labels = None
else:
    raw = load_demo(int(days), int(seed))
    DATA = raw[["timestamp", "demand_kwh", "temperature_c"]]
    KEY = f"demo-{days}-{seed}"
    SOURCE_WORD = "synthetic"
    raw_labels = raw["injected_anomaly"]

with st.spinner("Training forecasting model and running the 24-hour backtest…"):
    try:
        EV = get_eval(KEY, DATA)
    except ValueError as exc:
        st.error(f"Could not evaluate this data: {exc}")
        st.stop()

with st.sidebar:
    lo, hi = DATA.timestamp.min().date(), DATA.timestamp.max().date()
    default_lo = max(lo, (pd.Timestamp(hi) - pd.Timedelta(days=59)).date())
    view = st.date_input("View window", (default_lo, hi), min_value=lo, max_value=hi,
                         help="Limits charts and tables on Command Center, Anomaly, Cost and Data pages.")
    if isinstance(view, tuple) and len(view) == 2:
        v_lo, v_hi = view
    else:
        v_lo, v_hi = lo, hi
    st.divider()
    st.caption("Theme: ⋮ menu → Settings → Theme")
    st.markdown(f"[zubairlabs.com]({WEBSITE}) · [GitHub]({REPO})")


def in_view(df, col="timestamp"):
    t = df[col]
    return df[(t >= pd.Timestamp(v_lo)) & (t < pd.Timestamp(v_hi) + pd.Timedelta(days=1))]


VIEW = in_view(DATA)
RF1 = EV.one_step.iloc[0]
BT = EV.backtest_summary


def hero():
    src = ("<span class='pill real'>Uploaded data: " + html.escape(ss.get("upload_name", "")) + "</span>"
           if use_upload else "<span class='pill warn'>⚠ Synthetic demonstration data</span>")
    st.markdown(f"""<div class='gp-hero'><div class='eyebrow'>ZubairLabs · Energy intelligence</div>
<h1>⚡ GridPulse AI</h1><p>Hourly energy demand forecasting · anomaly screening · tariff cost insight</p>
{src}<span class='pill'>{len(DATA):,} hourly readings</span>
<span class='pill'>{DATA.timestamp.min():%d %b %Y} → {DATA.timestamp.max():%d %b %Y}</span>
<span class='pill'>Model: Random Forest + recursive 24h</span></div>""", unsafe_allow_html=True)
    if not use_upload:
        st.warning("**Synthetic data, not a real grid or meter.** Readings come from a documented simulation of a "
                   "commercial site. Forecasts are model estimates; nothing here is validated for grid operations, "
                   "billing or trading.", icon="⚠️")


def footer():
    st.markdown(f"<div class='footer'>© 2026 ZubairLabs · GridPulse AI · {'uploaded' if use_upload else 'synthetic demonstration'} data · "
                f"no paid APIs · <a href='{WEBSITE}'>zubairlabs.com</a></div>", unsafe_allow_html=True)


# ------------------------------------------------------------------ pages
def page_command():
    hero()
    section("Energy Command Center", "Consumption trends, peaks and patterns for the selected view window.")
    v = VIEW
    if v.empty:
        st.info("No readings in the selected window.")
        return
    prev = DATA[(DATA.timestamp < v.timestamp.min()) &
                (DATA.timestamp >= v.timestamp.min() - (v.timestamp.max() - v.timestamp.min()) - pd.Timedelta(hours=1))]
    change = (v.demand_kwh.mean() / prev.demand_kwh.mean() - 1) if len(prev) > 24 else None
    peak_row = v.loc[v.demand_kwh.idxmax()]
    load_factor = v.demand_kwh.mean() / v.demand_kwh.max()
    kpis([
        ("Total consumption", f"{v.demand_kwh.sum() / 1000:,.1f} MWh", f"{len(v):,} hours in view"),
        ("Average demand", f"{v.demand_kwh.mean():,.1f} kW", f"{change:+.1%} vs previous period" if change is not None else "no earlier period"),
        ("Peak demand", f"{peak_row.demand_kwh:,.1f} kW", f"{peak_row.timestamp:%d %b %H:%M}"),
        ("Load factor", f"{load_factor:.0%}", "average ÷ peak"),
        ("Day-ahead forecast MAE", f"{BT.iloc[0].MAE:.2f} kWh" if len(BT) else "n/a", "recursive backtest, assumed temp."),
    ])
    st.caption("Hourly kWh equals average kW over that hour, so the two units are used interchangeably here.")

    daily = v.set_index("timestamp").resample("D").agg(kwh=("demand_kwh", "sum"), peak=("demand_kwh", "max"),
                                                      temp=("temperature_c", "mean")).reset_index()
    fig = go.Figure()
    fig.add_bar(x=daily.timestamp, y=daily.kwh / 1000, name="Daily energy (MWh)", marker_color=ACTUAL, opacity=.85)
    fig.add_scatter(x=daily.timestamp, y=daily.temp, name="Mean temperature (°C)", yaxis="y2",
                    line=dict(color=AMBER, width=2))
    fig.update_layout(title=f"Daily consumption and temperature ({SOURCE_WORD} history)",
                      yaxis=dict(title="MWh per day"), yaxis2=dict(title="°C", overlaying="y", side="right", showgrid=False, tickformat=".0f"),
                      legend=dict(orientation="h", y=1.08, x=1, xanchor="right"))
    chart(fig, 400)

    c1, c2 = st.columns(2)
    with c1:
        prof = v.assign(hour=v.timestamp.dt.hour, day=np.where(v.timestamp.dt.dayofweek < 5, "Weekday", "Weekend"))
        p = prof.groupby(["day", "hour"], as_index=False).demand_kwh.mean()
        fig = px.line(p, x="hour", y="demand_kwh", color="day", markers=True,
                      color_discrete_map={"Weekday": CYAN, "Weekend": EMERALD},
                      title="Average daily profile", labels={"demand_kwh": "kW", "hour": "Hour of day"})
        chart(fig, 360)
    with c2:
        hm = v.assign(hour=v.timestamp.dt.hour, weekday=v.timestamp.dt.day_name().str[:3])
        piv = hm.pivot_table(index="weekday", columns="hour", values="demand_kwh", aggfunc="mean")
        piv = piv.reindex(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        fig = px.imshow(piv, aspect="auto", color_continuous_scale=["#041026", "#0E7490", CYAN, "#ECFEFF"],
                        labels=dict(color="kW", x="Hour", y=""), title="Demand heatmap: weekday × hour")
        chart(fig, 360)

    c3, c4 = st.columns(2)
    with c3:
        s = v.sample(min(len(v), 4000), random_state=1)
        fig = px.scatter(s, x="temperature_c", y="demand_kwh", opacity=.35, color_discrete_sequence=[BLUE],
                         title="Temperature sensitivity", labels={"temperature_c": "°C", "demand_kwh": "kW"})
        chart(fig, 340)
    with c4:
        dur = np.sort(v.demand_kwh.to_numpy())[::-1]
        fig = px.area(x=np.arange(1, len(dur) + 1) / len(dur) * 100, y=dur, color_discrete_sequence=[EMERALD],
                      labels={"x": "% of hours at or above", "y": "kW"}, title="Load duration curve")
        chart(fig, 340)

    wk = v.assign(we=v.timestamp.dt.dayofweek >= 5)
    weekday_avg, weekend_avg = wk[~wk.we].demand_kwh.mean(), wk[wk.we].demand_kwh.mean()
    peak_hour = int(v.groupby(v.timestamp.dt.hour).demand_kwh.mean().idxmax())
    note(f"<b>In this {SOURCE_WORD} data:</b> average weekday demand is {weekday_avg:,.1f} kW vs {weekend_avg:,.1f} kW at weekends; "
         f"the busiest hour on average starts at {peak_hour:02d}:00. These are descriptive statistics, not forecasts.")
    footer()


def page_forecast():
    hero()
    section("Demand Forecasting", "Recursive forecast of the next hours after the last reading, with recent history for comparison.")
    c1, c2, c3 = st.columns([1.2, 1.4, 1])
    horizon = c1.select_slider("Forecast horizon (hours)", [6, 12, 24, 48, 72], value=24)
    mode = c2.selectbox("Temperature assumption", list(TEMP_MODES), format_func=TEMP_MODES.get,
                        help="Future temperature is not known. The forecast uses this assumption, built only from history.")
    offset = c3.slider("Temperature adjustment (°C)", -8.0, 8.0, 0.0, 0.5,
                       help="Scenario: shift the assumed temperatures up or down.")
    fc = get_forecast(KEY, horizon, mode, float(offset), DATA, EV.model)

    ints = EV.intervals
    if len(ints):
        q = ints.set_index("horizon")
        hz = np.clip(np.arange(1, horizon + 1), 1, int(q.index.max()))
        fc["low"] = np.maximum(fc.forecast_kwh + q.loc[hz, "q10"].to_numpy(), 0)
        fc["high"] = fc.forecast_kwh + q.loc[hz, "q90"].to_numpy()

    hist = DATA.tail(max(72, 2 * horizon))
    last_week = DATA.set_index("timestamp").demand_kwh
    fc["same_hour_last_week"] = [last_week.get(t - pd.Timedelta(days=7), np.nan) for t in fc.timestamp]

    kpis([
        ("Forecast energy", f"{fc.forecast_kwh.sum() / 1000:,.2f} MWh", f"next {horizon} h"),
        ("Forecast peak", f"{fc.forecast_kwh.max():,.1f} kW", f"{fc.loc[fc.forecast_kwh.idxmax(), 'timestamp']:%a %d %b %H:%M}"),
        ("Same hours last week", f"{fc.same_hour_last_week.sum() / 1000:,.2f} MWh",
         f"{fc.forecast_kwh.sum() / fc.same_hour_last_week.sum() - 1:+.1%} forecast vs last week" if fc.same_hour_last_week.notna().all() else "n/a"),
        ("Typical day-ahead error", f"±{BT.iloc[0].MAE:.1f} kWh" if len(BT) else "n/a", "backtest MAE per hour"),
    ])
    legend_keys(model=False)

    fig = go.Figure()
    fig.add_scatter(x=hist.timestamp, y=hist.demand_kwh, name=f"Actual ({SOURCE_WORD})", line=dict(color=ACTUAL, width=2))
    if "low" in fc:
        fig.add_scatter(x=list(fc.timestamp) + list(fc.timestamp[::-1]), y=list(fc.high) + list(fc.low[::-1]),
                        fill="toself", fillcolor="rgba(16,185,129,.16)", line=dict(width=0), name="80% empirical band",
                        hoverinfo="skip")
    fig.add_scatter(x=fc.timestamp, y=fc.forecast_kwh, name="Forecast (model)", line=dict(color=FORECAST, width=3, dash="dash"),
                    mode="lines+markers", marker=dict(size=5))
    fig.add_scatter(x=fc.timestamp, y=fc.same_hour_last_week, name="Same hour last week (actual)",
                    line=dict(color=SLATE, width=1.5, dash="dot"))
    fig.add_vline(x=DATA.timestamp.iloc[-1], line=dict(color=SLATE, dash="dot"))
    fig.add_annotation(x=DATA.timestamp.iloc[-1], y=1.02, yref="paper", text="last reading", showarrow=False, font=dict(size=11))
    fig.update_layout(title="Recent history and the forecast", yaxis_title="kW",
                      legend=dict(orientation="h", y=1.1, x=1, xanchor="right"))
    chart(fig, 440)

    c1, c2 = st.columns([1.3, 1])
    with c1:
        fig = go.Figure()
        fig.add_scatter(x=fc.timestamp, y=fc.assumed_temperature_c, name="Assumed temperature", line=dict(color=AMBER, dash="dash"))
        fig.add_scatter(x=DATA.timestamp.tail(48), y=DATA.temperature_c.tail(48), name="Measured temperature", line=dict(color=AMBER))
        fig.update_layout(title="Temperature: measured history vs forecast assumption", yaxis_title="°C",
                          legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))
        chart(fig, 320)
    with c2:
        st.markdown("**How this forecast is made**")
        st.markdown(
            "- Model: Random Forest trained on **all** history (features in Model Performance).\n"
            "- Each hour uses demand 24 h and 168 h earlier (observed) and the mean of the previous 24 h. "
            "After the first hour that mean includes the model's own predictions; **actual future values are never used**.\n"
            f"- Temperature: *{TEMP_MODES[mode].lower()}*{f', shifted {offset:+.1f} °C' if offset else ''}.\n"
            "- The shaded band is the 10th–90th percentile of past backtest errors at each horizon"
            + (" (beyond 24 h it reuses the 24 h band)" if horizon > 24 else "") + ". It is an empirical guide, not a guarantee.")
        if horizon > 24:
            st.warning("Beyond 24 hours the lag inputs themselves become predictions and errors compound. "
                       "The backtest only measures 24-hour accuracy.")
    with st.expander("Forecast table"):
        show = fc.rename(columns={"forecast_kwh": "forecast_kwh (model)", "assumed_temperature_c": "assumed_temperature_c (assumption)"})
        st.dataframe(rnd(show, 2), width="stretch", hide_index=True)
    st.download_button("Download forecast CSV", csv_bytes(rnd(fc, 3)), "gridpulse_forecast.csv", "text/csv",
                       key="dl_forecast")
    footer()


def page_anomaly():
    hero()
    section("Anomaly Intelligence", "Hours whose demand differs unusually from what is typical for that hour, day type and temperature.")
    c1, c2 = st.columns([1, 2])
    rate = c1.slider("Share of hours to flag", 0.005, 0.10, 0.02, 0.005, format="%.3f",
                     help="Isolation Forest contamination: the fraction of hours treated as outliers.")
    flags = get_flags(KEY, float(rate), DATA)
    v = in_view(flags)
    f = v[v.anomaly]
    kinds = f.kind.value_counts()
    kpis([
        ("Flagged hours (view)", f"{len(f):,}", f"of {len(v):,} hours"),
        ("Higher than expected", f"{kinds.get('Higher than expected', 0):,}", "spikes"),
        ("Lower than expected", f"{kinds.get('Lower than expected', 0):,}", "dips / outages"),
        ("Flat / stuck readings", f"{kinds.get('Flat / stuck reading', 0):,}", "constant values"),
        ("Excess energy in flags", f"{f.residual_kwh.clip(lower=0).sum():,.0f} kWh", "above expected"),
    ])
    if raw_labels is not None:
        sc = anomaly_scores(flags, raw_labels)
        c2.markdown(f"<div class='note'><b>Measured on the synthetic data's injected anomalies</b> (whole history): "
                    f"precision {sc['precision']:.0%}, recall {sc['recall']:.0%} — {sc['tp']} of {sc['labelled']} injected "
                    f"hours caught, {sc['fp']} other hours flagged. Real meters have no such answer key.</div>",
                    unsafe_allow_html=True)
    else:
        c2.info("Uploaded data has no known anomaly labels, so detection accuracy cannot be measured here.")

    fig = go.Figure()
    fig.add_scatter(x=v.timestamp, y=v.demand_kwh, name=f"Actual ({SOURCE_WORD})", line=dict(color=ACTUAL, width=1.4))
    fig.add_scatter(x=v.timestamp, y=v.expected_kwh, name="Expected (typical for hour/day/temperature)",
                    line=dict(color=SLATE, width=1, dash="dot"))
    colors = {"Higher than expected": RED, "Lower than expected": AMBER, "Flat / stuck reading": "#A855F7"}
    for k, col in colors.items():
        g = f[f.kind == k]
        if len(g):
            fig.add_scatter(x=g.timestamp, y=g.demand_kwh, mode="markers", name=k, marker=dict(color=col, size=9, line=dict(width=1, color="white")))
    fig.update_layout(title="Demand with flagged hours", yaxis_title="kW", legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))
    chart(fig, 430)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(v.assign(status=np.where(v.anomaly, "Flagged", "Normal")), x="residual_kwh", nbins=80, color="status",
                           color_discrete_map={"Normal": CYAN, "Flagged": RED},
                           title="Residual distribution (actual − expected)", labels={"residual_kwh": "kWh", "status": ""})
        chart(fig, 330)
    with c2:
        by_h = v.groupby(v.timestamp.dt.hour).anomaly.sum().reset_index()
        by_h.columns = ["hour", "flags"]
        fig = px.bar(by_h, x="hour", y="flags", color_discrete_sequence=[EMERALD], title="Flags by hour of day",
                     labels={"hour": "Hour", "flags": "Flagged hours"})
        chart(fig, 330)

    st.markdown("**Flagged hours** (highest anomaly score first)")
    cols = ["timestamp", "demand_kwh", "expected_kwh", "residual_kwh", "temperature_c", "kind", "anomaly_score"]
    st.dataframe(f.sort_values("anomaly_score", ascending=False)[cols].pipe(rnd, 2), width="stretch", hide_index=True,
                 column_config={"anomaly_score": st.column_config.ProgressColumn("score", min_value=0.0, max_value=1.0, format="%.2f")})
    st.download_button("Download flagged hours CSV", csv_bytes(rnd(f[cols], 3)), "gridpulse_anomalies.csv", "text/csv",
                       key="dl_anom")
    st.caption("Isolation Forest flags statistical outliers for review. A flag is not a confirmed fault, theft or meter error. "
               "The expected value uses neighbouring weeks on both sides, so this is a retrospective review, not real-time alerting.")
    footer()


def page_cost():
    hero()
    section("Cost Intelligence", "Illustrative cost estimates under a tariff you configure. Not a utility bill.")
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        cur = c1.text_input("Currency label", "USD", max_chars=6)
        mode = c2.radio("Tariff type", ["time_of_use", "flat"], format_func={"time_of_use": "Time of use", "flat": "Flat rate"}.get,
                        horizontal=True)
        demand_charge = c3.number_input("Demand charge per kW of monthly peak", 0.0, 100.0, 8.0, 0.5)
        fixed = c4.number_input("Fixed charge per month", 0.0, 100000.0, 50.0, 10.0)
        if mode == "flat":
            flat = st.number_input("Energy rate per kWh", 0.0, 5.0, 0.18, 0.01)
            tariff = Tariff(currency=cur, mode="flat", flat_rate=flat, demand_charge=demand_charge, fixed_monthly=fixed)
        else:
            d1, d2, d3, d4 = st.columns(4)
            peak = d1.number_input("Peak rate per kWh", 0.0, 5.0, 0.26, 0.01)
            off = d2.number_input("Off-peak rate per kWh", 0.0, 5.0, 0.12, 0.01)
            window = d3.slider("Peak hours", 0, 24, (8, 20))
            we = d4.toggle("Weekends are off-peak", True)
            tariff = Tariff(currency=cur, mode="time_of_use", peak_rate=peak, offpeak_rate=off, peak_start=window[0],
                            peak_end=window[1], weekend_offpeak=we, demand_charge=demand_charge, fixed_monthly=fixed)
    try:
        tariff.validate()
    except ValueError as exc:
        st.error(str(exc))
        return
    cur = tariff.currency or ""
    v = VIEW
    hc = energy_cost(v, tariff)
    bills = monthly_bill(v, tariff)
    fc = get_forecast(KEY, 24, "repeat_last_day", 0.0, DATA, EV.model)
    fcc = energy_cost(fc.rename(columns={"forecast_kwh": "demand_kwh"}), tariff)
    days_in_view = max(len(v) / 24, 1)
    peak_share = hc.loc[hc.period == "Peak", "demand_kwh"].sum() / hc.demand_kwh.sum() if mode == "time_of_use" else None
    kpis([
        ("Energy cost (view)", fmt_money(hc.energy_cost.sum(), cur), f"{days_in_view:,.0f} days"),
        ("Average per day", fmt_money(hc.energy_cost.sum() / days_in_view, cur), "energy only"),
        ("Effective rate", f"{hc.energy_cost.sum() / hc.demand_kwh.sum():.3f}", f"{cur} per kWh (energy only)"),
        ("Peak-period share", f"{peak_share:.0%}" if peak_share is not None else "n/a", "of kWh"),
        ("Next 24 h (forecast)", fmt_money(fcc.energy_cost.sum(), cur), "model estimate × tariff"),
    ])

    c1, c2 = st.columns([1.4, 1])
    with c1:
        b = bills.melt(id_vars=["month"], value_vars=["energy_cost", "demand_charge", "fixed"], var_name="component", value_name="cost")
        b["component"] = b.component.map({"energy_cost": "Energy", "demand_charge": "Demand charge", "fixed": "Fixed"})
        fig = px.bar(b, x="month", y="cost", color="component", title="Illustrative monthly cost by component",
                     color_discrete_map={"Energy": CYAN, "Demand charge": AMBER, "Fixed": SLATE}, labels={"cost": cur, "month": ""})
        chart(fig, 380)
        if bills.partial_month.any():
            st.caption("Months marked partial in the table are only partly inside the view window.")
    with c2:
        by_h = hc.groupby(hc.timestamp.dt.hour).agg(cost=("energy_cost", "mean")).reset_index().rename(columns={"timestamp": "hour"})
        if mode == "time_of_use":
            in_peak = (by_h.hour >= tariff.peak_start) & (by_h.hour < tariff.peak_end)
            by_h["period"] = np.where(in_peak, "Peak window" + (" (weekdays)" if tariff.weekend_offpeak else ""), "Off-peak")
        else:
            by_h["period"] = "Flat"
        fig = px.bar(by_h, x="hour", y="cost", color="period", title="Average energy cost by hour of day",
                     color_discrete_sequence=[EMERALD, RED, CYAN], labels={"hour": "Hour", "cost": cur, "period": ""})
        chart(fig, 380)

    if mode == "time_of_use":
        st.markdown("**What-if: shift flexible load out of the peak window**")
        s1, s2 = st.columns([1, 2])
        share = s1.slider("Share of peak-period energy moved to off-peak", 0, 30, 10, 1, format="%d%%") / 100
        moved_kwh = hc.loc[hc.period == "Peak", "demand_kwh"].sum() * share
        saving = moved_kwh * (tariff.peak_rate - tariff.offpeak_rate)
        s2.markdown(f"<div class='note'>Moving {share:.0%} of peak energy ({moved_kwh:,.0f} kWh in this window) to off-peak hours "
                    f"would change the energy cost by <b>{fmt_money(-saving, cur)}</b> under this tariff. This is simple arithmetic on "
                    f"{SOURCE_WORD} data; it ignores rebound, comfort, process limits and demand charges.</div>", unsafe_allow_html=True)

    st.markdown("**Monthly table**")
    st.dataframe(rnd(bills, 2), width="stretch", hide_index=True,
                 column_config={"partial_month": st.column_config.CheckboxColumn("partial month?")})
    st.download_button("Download monthly cost CSV", csv_bytes(rnd(bills, 3)), "gridpulse_monthly_cost.csv", "text/csv", key="dl_cost")
    st.caption("Estimates multiply kWh by the rates entered above. Real bills include taxes, network charges, "
               "seasonal tariffs and metering rules that are not modelled.")
    footer()


def page_data():
    hero()
    section("Data Explorer", "Upload your own hourly readings, check data quality, and export.")
    with st.container(border=True):
        st.markdown(f"**Upload a CSV** with columns `timestamp`, `demand_kwh`, `temperature_c` — hourly, at least "
                    f"{MIN_HOURS // 24} days, at most {MAX_HOURS:,} rows, {MAX_UPLOAD_MB} MB. Files stay in your browser session only.")
        u1, u2 = st.columns([2, 1])
        up = u1.file_uploader("Hourly CSV", type=["csv"], label_visibility="collapsed")
        repair = u2.toggle("Repair small issues", True,
                           help="Merge duplicate timestamps and fill gaps of up to 3 hours by interpolation.")
        template = load_demo(30, 7)[["timestamp", "demand_kwh", "temperature_c"]]
        u2.download_button("Download a template CSV", csv_bytes(template), "gridpulse_template.csv", "text/csv", key="dl_tpl")
        if up is not None:
            content = up.getvalue()
            if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
                st.error(f"File is larger than {MAX_UPLOAD_MB} MB.")
            else:
                try:
                    df = pd.read_csv(io.BytesIO(content), nrows=MAX_HOURS + 1)
                except Exception as exc:  # noqa: BLE001 - show parse errors to the user
                    st.error(f"Could not read this file as CSV: {exc}")
                    df = None
                if df is not None:
                    clean, rep = validate_report(df, repair=repair)
                    if clean is None:
                        st.error("This file can't be used yet:\n\n- " + "\n- ".join(rep.problems))
                    else:
                        h = hashlib.sha256(content + (b"r" if repair else b"s")).hexdigest()[:16]
                        msg = f"Valid: {rep.rows_out:,} hourly readings, {rep.start} → {rep.end}."
                        if rep.duplicates_merged or rep.gaps_filled:
                            msg += f" Repaired: {rep.duplicates_merged} duplicate(s) merged, {rep.gaps_filled} hour(s) interpolated."
                        st.success(msg)
                        if ss.get("upload_hash") != h:
                            if st.button("Analyse this file", type="primary"):
                                ss.update(upload_df=clean, upload_name=up.name[:60], upload_hash=h, use_upload=True)
                                ss.pop("source_choice", None)
                                st.rerun()
                        else:
                            st.caption("This file is the active data source.")
        if "upload_df" in ss and st.button("Remove uploaded data and use the synthetic demo"):
            for k in ("upload_df", "upload_name", "upload_hash", "use_upload", "source_choice"):
                ss.pop(k, None)
            st.rerun()

    st.markdown("**Data quality**")
    gaps = int((DATA.timestamp.diff().dropna() != pd.Timedelta(hours=1)).sum())
    kpis([
        ("Rows", f"{len(DATA):,}", "hourly readings"),
        ("Period", f"{(DATA.timestamp.max() - DATA.timestamp.min()).days + 1:,} days", f"{DATA.timestamp.min():%d %b %Y} start"),
        ("Gaps / missing", f"{gaps} / {int(DATA.isna().sum().sum())}", "after validation"),
        ("Demand range", f"{DATA.demand_kwh.min():,.0f} to {DATA.demand_kwh.max():,.0f}", "kW, minimum to maximum"),
        ("Temperature range", f"{DATA.temperature_c.min():.1f} to {DATA.temperature_c.max():.1f}", "°C, minimum to maximum"),
    ])

    st.markdown("**What the model can use at forecast time**")
    rows = [{"Column / feature": k, "Known at forecast time?": "Yes" if k in ("hour", "weekday", "lag24", "lag168") else
             ("Assumed (not known)" if k == "temperature_c" else "Partly predicted after hour 1"), "Meaning": v}
            for k, v in FEATURE_INFO.items()]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(VIEW, x="demand_kwh", nbins=60, color_discrete_sequence=[CYAN], title="Demand distribution (view)",
                           labels={"demand_kwh": "kW"})
        chart(fig, 320)
    with c2:
        m = VIEW.assign(month=VIEW.timestamp.dt.to_period("M").astype(str)).groupby("month", as_index=False).agg(
            mwh=("demand_kwh", lambda s: s.sum() / 1000))
        fig = px.bar(m, x="month", y="mwh", color_discrete_sequence=[EMERALD], title="Monthly energy (view)", labels={"mwh": "MWh", "month": ""})
        chart(fig, 320)

    st.markdown(f"**Readings in view** ({len(VIEW):,} rows, {SOURCE_WORD})")
    st.dataframe(rnd(VIEW, 3), width="stretch", hide_index=True, height=320)
    d1, d2, d3 = st.columns(3)
    d1.download_button("Download readings in view (CSV)", csv_bytes(rnd(VIEW, 3)), "gridpulse_readings_view.csv", "text/csv", key="dl_view")
    full = DATA if use_upload else load_demo(int(days), int(seed))
    d2.download_button("Download full dataset (CSV)", csv_bytes(rnd(full, 3)), "gridpulse_dataset.csv", "text/csv", key="dl_full")
    d3.download_button("Download evaluation (JSON)", evaluation_json(EV).encode(), "gridpulse_evaluation.json", "application/json", key="dl_json")
    if not use_upload:
        st.caption("The full synthetic dataset includes `injected_anomaly`, the answer key used to score the anomaly screen.")
    footer()


def page_model():
    hero()
    section("Model Performance", "Measured on the latest 20% of the history, which the evaluation model never saw.")
    m = EV.meta
    rf_bt = BT.iloc[0] if len(BT) else None
    naive_best = BT.iloc[2:].sort_values("MAE").iloc[0] if len(BT) else None
    kpis([
        ("Day-ahead MAE", f"{rf_bt.MAE:.2f} kWh" if rf_bt is not None else "n/a", "recursive 24 h, assumed temperature"),
        ("Day-ahead RMSE", f"{rf_bt.RMSE:.2f} kWh" if rf_bt is not None else "n/a", f"MAPE {rf_bt.MAPE:.1f}%" if rf_bt is not None else ""),
        ("Best naive MAE", f"{naive_best.MAE:.2f} kWh" if naive_best is not None else "n/a",
         naive_best.Method.replace("Seasonal naive: ", "") if naive_best is not None else ""),
        ("One-step MAE", f"{RF1.MAE:.2f} kWh", f"RMSE {RF1.RMSE:.2f}"),
        ("Backtest days", f"{m['backtest_origins']}", f"test from {m['test_period'][0][:10]}"),
    ])
    if rf_bt is not None and naive_best is not None:
        rel = 1 - rf_bt.MAE / naive_best.MAE
        note(f"On this {SOURCE_WORD} test period the recursive model's day-ahead MAE is <b>{rel:.0%} lower</b> than the best seasonal-naive "
             f"baseline. With the actual future temperature (unknowable in practice) it would be {BT.iloc[1].MAE:.2f} kWh — "
             "the gap shows how much the temperature assumption costs.")

    st.markdown("**Day-ahead recursive backtest** — one forecast per day at 23:00 for the next 24 hours")
    st.dataframe(rnd(BT, 3), width="stretch", hide_index=True)
    st.markdown("**One-step-ahead holdout** — every input observed; easier than the real forecasting task")
    st.dataframe(rnd(EV.one_step, 3), width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        bh = EV.by_horizon.rename(columns={"MAE": "Random Forest (recursive)", "MAE naive 24h": "Same hour yesterday",
                                           "MAE naive 168h": "Same hour last week"})
        bh = bh.melt(id_vars="horizon", var_name="method", value_name="MAE")
        fig = px.line(bh, x="horizon", y="MAE", color="method", markers=True, title="Error by forecast horizon",
                      color_discrete_map={"Random Forest (recursive)": EMERALD, "Same hour yesterday": SLATE, "Same hour last week": BLUE},
                      labels={"horizon": "Hours ahead", "MAE": "MAE (kWh)"})
        chart(fig, 360)
    with c2:
        imp = EV.importance.sort_values("mae_increase")
        fig = px.bar(imp, x="mae_increase", y="feature", orientation="h", error_x="std", color_discrete_sequence=[CYAN],
                     title="Permutation importance (MAE increase when shuffled)", labels={"mae_increase": "kWh", "feature": ""})
        chart(fig, 360)

    hold = EV.holdout.tail(24 * 14)
    fig = go.Figure()
    fig.add_scatter(x=hold.timestamp, y=hold.demand_kwh, name=f"Actual ({SOURCE_WORD})", line=dict(color=ACTUAL, width=2))
    fig.add_scatter(x=hold.timestamp, y=hold.predicted_kwh, name="Model, one step ahead", line=dict(color=MODEL, width=1.6, dash="dot"))
    fig.update_layout(title="Holdout: last 14 days, actual vs one-step model estimate", yaxis_title="kW",
                      legend=dict(orientation="h", y=1.1, x=1, xanchor="right"))
    chart(fig, 380)

    bt = EV.backtest
    if len(bt):
        last_origins = bt.origin.drop_duplicates().tail(5)
        b = bt[bt.origin.isin(last_origins)]
        fig = go.Figure()
        fig.add_scatter(x=b.timestamp, y=b.actual, name=f"Actual ({SOURCE_WORD})", line=dict(color=ACTUAL, width=2))
        for o, g in b.groupby("origin"):
            fig.add_scatter(x=g.timestamp, y=g.forecast, name=f"Forecast from {o:%d %b}", line=dict(color=FORECAST, dash="dash"),
                            showlegend=bool(o == last_origins.iloc[0]))
        fig.update_layout(title="Last 5 day-ahead backtest forecasts vs actual", yaxis_title="kW",
                          legend=dict(orientation="h", y=1.1, x=1, xanchor="right"))
        chart(fig, 360)
        c3, c4 = st.columns(2)
        with c3:
            err = bt.assign(hour=bt.timestamp.dt.hour).groupby("hour").error.agg(lambda e: np.mean(np.abs(e))).reset_index()
            fig = px.bar(err, x="hour", y="error", color_discrete_sequence=[BLUE], title="Backtest MAE by hour of day",
                         labels={"hour": "Hour", "error": "MAE (kWh)"})
            chart(fig, 320)
        with c4:
            fig = px.scatter(bt.sample(min(len(bt), 3000), random_state=1), x="actual", y="forecast", opacity=.4,
                             color_discrete_sequence=[EMERALD], title="Backtest forecast vs actual",
                             labels={"actual": "Actual kW", "forecast": "Forecast kW"})
            lim = [bt[["actual", "forecast"]].min().min(), bt[["actual", "forecast"]].max().max()]
            fig.add_scatter(x=lim, y=lim, mode="lines", line=dict(color=SLATE, dash="dot"), name="perfect")
            chart(fig, 320)

    with st.expander("Methodology and assumptions"):
        st.markdown(f"""
- **Split:** chronological. Training {m['train_period'][0][:16]} → {m['train_period'][1][:16]} ({m['train_rows']:,} rows);
  test {m['test_period'][0][:16]} → {m['test_period'][1][:16]} ({m['test_rows']:,} rows). The first week is used only for lags.
- **Model:** `RandomForestRegressor(n_estimators=120, max_depth=14, min_samples_leaf=3, random_state=42)` on
  hour, weekday, temperature, demand 24 h and 168 h earlier, and the mean of the previous 24 h.
- **One-step holdout:** each test hour is predicted with observed lags — an optimistic view.
- **Day-ahead backtest:** at 23:00 each test day, the training-period model forecasts the next 24 hours recursively,
  using only data up to that moment and the temperature assumption *{TEMP_MODES[m['temp_mode']].lower()}*.
  The same run with actual temperatures is shown as an upper bound.
- **Baselines:** same hour yesterday and same hour last week, both known at forecast time.
- **Live forecast:** a model refit on all rows.
- **Not modelled:** holidays, occupancy changes, tariffs' effect on behaviour, equipment changes, weather forecast error beyond
  the persistence assumption.""")
    st.download_button("Download evaluation (JSON)", evaluation_json(EV).encode(), "gridpulse_evaluation.json", "application/json",
                       key="dl_json_model")
    footer()


pages = [
    st.Page(page_command, title="Energy Command Center", icon=":material/dashboard:", url_path="command", default=True),
    st.Page(page_forecast, title="Demand Forecasting", icon=":material/trending_up:", url_path="forecast"),
    st.Page(page_anomaly, title="Anomaly Intelligence", icon=":material/crisis_alert:", url_path="anomalies"),
    st.Page(page_cost, title="Cost Intelligence", icon=":material/payments:", url_path="cost"),
    st.Page(page_data, title="Data Explorer", icon=":material/table_view:", url_path="data"),
    st.Page(page_model, title="Model Performance", icon=":material/monitoring:", url_path="model"),
]
st.navigation(pages, position="sidebar").run()
