"""GridPulse AI core: synthetic data, validation, forecasting, anomaly screening and cost.

Design rules
------------
* Every model feature for hour ``t`` uses only demand observed **before** ``t``
  (lags of 24h and 168h, and the mean of the previous 24 hours) plus calendar
  values and temperature for ``t``.
* At forecast time future temperature is unknown. The forecast uses an explicit
  temperature assumption (by default: repeat the temperatures of the last 24
  observed hours) and never reads actual future values.
* Recursive forecasts feed their own predictions back as inputs; tests check
  that changing future actuals cannot change a forecast.
* Data are synthetic unless the user uploads their own. Metrics describe
  performance on that data only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error

REQUIRED = ["timestamp", "demand_kwh", "temperature_c"]
FEATURES = ["hour", "weekday", "temperature_c", "lag24", "lag168", "mean24"]
FEATURE_INFO = {
    "hour": "Hour of day (calendar, known in advance)",
    "weekday": "Day of week, 0 = Monday (calendar, known in advance)",
    "temperature_c": "Temperature for the hour (history: measured; forecast: assumption)",
    "lag24": "Demand 24 hours earlier (observed)",
    "lag168": "Demand 168 hours (one week) earlier (observed)",
    "mean24": "Mean demand over the previous 24 hours (observed or, in a recursive forecast, partly predicted)",
}
MIN_HOURS = 360          # 15 days: one week for lags plus enough rows to train and test
MAX_HOURS = 3 * 8784     # three years of hourly data
RANDOM_STATE = 42
TEST_SHARE = 0.20
HORIZON = 24
TEMP_MODES = {
    "repeat_last_day": "Repeat the last 24 observed hourly temperatures",
    "hold_last": "Hold the last observed temperature constant (original behaviour)",
}


# --------------------------------------------------------------------------- data
def demo(days: int = 120, seed: int = 42, start: str = "2025-01-01", anomalies: bool = True) -> pd.DataFrame:
    """Synthetic hourly electricity demand for a mid-sized commercial site.

    Demand combines an occupancy profile (weekday vs weekend), heating below 14 °C,
    cooling above 22 °C, a slow upward trend and autocorrelated noise. Temperature
    has annual and daily cycles plus multi-day weather swings. When ``anomalies``
    is true a few labelled events are injected (spikes, dips, stuck meter) so the
    anomaly screen can be measured against a known answer.
    """
    if days < 1:
        raise ValueError("days must be positive")
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, periods=days * 24, freq="h")
    n = len(t)
    h = t.hour.to_numpy()
    wd = t.dayofweek.to_numpy()
    doy = t.dayofyear.to_numpy()

    # temperature: annual cycle + daily cycle + AR(1) daily weather + hourly noise
    daily = np.zeros(days)
    for i in range(1, days):
        daily[i] = 0.8 * daily[i - 1] + rng.normal(0, 1.6)
    temp = (17 + 10 * np.sin(2 * np.pi * (doy - 110) / 365.25)
            + 4.5 * np.sin(2 * np.pi * (h - 9) / 24)
            + np.repeat(daily, 24) + rng.normal(0, 0.7, n))

    occupancy = np.exp(-(((h - 13) / 4.5) ** 2))
    weekday = wd < 5
    profile = np.where(weekday, 48 * occupancy, 14 * occupancy)
    cooling = 3.2 * np.maximum(temp - 22, 0)
    heating = 2.4 * np.maximum(14 - temp, 0)
    trend = 1 + 0.03 * np.arange(n) / 8760
    noise = np.zeros(n)
    eps = rng.normal(0, 4.5, n)
    for i in range(1, n):
        noise[i] = 0.5 * noise[i - 1] + eps[i]
    demand = (95 + profile + cooling + heating) * trend + noise

    labels = np.zeros(n, dtype=bool)
    if anomalies and days >= 14:
        n_events = max(2, days // 6)
        starts = rng.choice(np.arange(24 * 8, n - 12), size=n_events, replace=False)
        for s in np.sort(starts):
            kind = rng.choice(["spike", "dip", "stuck"], p=[0.45, 0.35, 0.20])
            if kind == "spike":
                length = int(rng.integers(1, 4))
                demand[s:s + length] *= rng.uniform(1.45, 1.9)
            elif kind == "dip":
                length = int(rng.integers(1, 4))
                demand[s:s + length] *= rng.uniform(0.25, 0.55)
            else:
                length = int(rng.integers(5, 10))
                demand[s:s + length] = demand[s]
            labels[s:s + length] = True

    return pd.DataFrame({
        "timestamp": t,
        "demand_kwh": np.round(np.maximum(demand, 0), 3),
        "temperature_c": np.round(temp, 2),
        "injected_anomaly": labels,
    })


@dataclass
class DataReport:
    rows_in: int = 0
    rows_out: int = 0
    start: str = ""
    end: str = ""
    duplicates_merged: int = 0
    gaps_filled: int = 0
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def as_dict(self) -> dict:
        return {"rows_in": self.rows_in, "rows_out": self.rows_out, "start": self.start, "end": self.end,
                "duplicates_merged": self.duplicates_merged, "gaps_filled": self.gaps_filled,
                "problems": list(self.problems)}


def _coerce(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    problems: list[str] = []
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a table (pandas DataFrame)")
    cols = {str(c).strip().lower(): c for c in df.columns}
    missing = [c for c in REQUIRED if c not in cols]
    if missing:
        raise ValueError("Required columns: timestamp, demand_kwh, temperature_c (missing: " + ", ".join(missing) + ")")
    d = pd.DataFrame({c: df[cols[c]] for c in REQUIRED})
    ts = pd.to_datetime(d["timestamp"], errors="coerce", utc=False)
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert(None)
    bad_ts = int(ts.isna().sum())
    if bad_ts:
        problems.append(f"{bad_ts} timestamp value(s) could not be parsed")
    d["timestamp"] = ts
    for c in ("demand_kwh", "temperature_c"):
        num = pd.to_numeric(d[c], errors="coerce")
        bad = int(num.isna().sum() - d[c].isna().sum())
        if bad:
            problems.append(f"{bad} non-numeric value(s) in {c}")
        d[c] = num
    return d, problems


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Strict validation: continuous hourly series with no missing or negative values.

    Raises ``ValueError`` describing the problem. Use :func:`validate_report` with
    ``repair=True`` to fix small, well-defined issues instead.
    """
    d, problems = _coerce(df)
    if problems:
        raise ValueError("; ".join(problems))
    d = d.sort_values("timestamp").reset_index(drop=True)
    if len(d) > MAX_HOURS:
        raise ValueError(f"Too many rows ({len(d):,}); the limit is {MAX_HOURS:,} hourly readings (3 years)")
    if (len(d) < MIN_HOURS or d.isna().any().any() or (d.demand_kwh < 0).any()
            or not d.timestamp.diff().dropna().eq(pd.Timedelta(hours=1)).all()):
        raise ValueError("Need 15+ days of continuous, nonnegative hourly readings without gaps or missing values")
    if ((d.temperature_c < -60) | (d.temperature_c > 60)).any():
        raise ValueError("temperature_c must be between -60 and 60 °C")
    return d


def validate_report(df: pd.DataFrame, repair: bool = False, max_gap_hours: int = 3) -> tuple[pd.DataFrame | None, DataReport]:
    """Validate with a readable report. With ``repair`` it merges duplicate timestamps
    (mean), fills gaps up to ``max_gap_hours`` by time interpolation, and drops nothing else."""
    rep = DataReport(rows_in=len(df) if isinstance(df, pd.DataFrame) else 0)
    try:
        d, problems = _coerce(df)
    except ValueError as e:
        rep.problems.append(str(e))
        return None, rep
    rep.problems.extend(problems)
    d = d.dropna(subset=["timestamp"]).sort_values("timestamp")
    if repair:
        dup = int(d.timestamp.duplicated().sum())
        if dup:
            d = d.groupby("timestamp", as_index=False)[["demand_kwh", "temperature_c"]].mean()
            rep.duplicates_merged = dup
        if len(d) >= 2:
            d["timestamp"] = d["timestamp"].dt.floor("h")
            d = d.groupby("timestamp", as_index=False)[["demand_kwh", "temperature_c"]].mean()
            full = pd.date_range(d.timestamp.min(), d.timestamp.max(), freq="h")
            d = d.set_index("timestamp").reindex(full)
            gap_mask = d.demand_kwh.isna() | d.temperature_c.isna()
            # only fill runs of missing values no longer than max_gap_hours
            run_id = (~gap_mask).cumsum()
            run_len = gap_mask.groupby(run_id).transform("sum")
            fillable = gap_mask & (run_len <= max_gap_hours)
            filled = d.interpolate(method="time", limit=max_gap_hours, limit_area="inside")
            d = d.where(~fillable, filled)
            rep.gaps_filled = int(fillable.sum())
            d = d.rename_axis("timestamp").reset_index()
        rep.problems = [p for p in rep.problems if "timestamp" not in p]  # unparsable rows were dropped
    try:
        clean = validate(d)
    except ValueError as e:
        msg = str(e)
        detail = []
        if len(d) and d.timestamp.duplicated().any():
            detail.append(f"{int(d.timestamp.duplicated().sum())} duplicate timestamp(s)")
        if len(d) > 1:
            gaps = d.timestamp.sort_values().diff().dropna()
            n_gaps = int((gaps > pd.Timedelta(hours=1)).sum())
            if n_gaps:
                detail.append(f"{n_gaps} gap(s) in the hourly sequence")
        if d[["demand_kwh", "temperature_c"]].isna().any().any():
            detail.append(f"{int(d[['demand_kwh', 'temperature_c']].isna().any(axis=1).sum())} row(s) with missing values")
        if (d.demand_kwh < 0).any():
            detail.append(f"{int((d.demand_kwh < 0).sum())} negative demand value(s)")
        if len(d) < MIN_HOURS:
            detail.append(f"only {len(d)} hourly rows (minimum {MIN_HOURS})")
        rep.problems.append(msg + (" — found: " + ", ".join(detail) if detail else ""))
        rep.rows_out = 0
        return None, rep
    rep.rows_out = len(clean)
    rep.start, rep.end = str(clean.timestamp.iloc[0]), str(clean.timestamp.iloc[-1])
    return clean, rep


# ----------------------------------------------------------------------- features
def features(d: pd.DataFrame) -> pd.DataFrame:
    """Past-only features. Rows without a full week of history are dropped."""
    d = d.copy()
    d["hour"] = d.timestamp.dt.hour
    d["weekday"] = d.timestamp.dt.dayofweek
    d["lag24"] = d.demand_kwh.shift(24)
    d["lag168"] = d.demand_kwh.shift(168)
    d["mean24"] = d.demand_kwh.shift(1).rolling(24).mean()
    return d.dropna(subset=["lag24", "lag168", "mean24"]).reset_index(drop=True)


def make_model(n_estimators: int = 120) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=n_estimators, max_depth=14, min_samples_leaf=3,
                                 n_jobs=-1, random_state=RANDOM_STATE)


def _split(f: pd.DataFrame, share: float = TEST_SHARE) -> int:
    return int(len(f) * (1 - share))


def fit(d: pd.DataFrame):
    """Chronological 80/20 one-step evaluation. Returns (model, {'MAE','RMSE'}, holdout frame)."""
    f = features(validate(d))
    cut = _split(f)
    train, test = f.iloc[:cut], f.iloc[cut:]
    m = make_model().fit(train[FEATURES], train.demand_kwh)
    p = m.predict(test[FEATURES])
    out = test[["timestamp", "demand_kwh"]].copy()
    out["predicted_kwh"] = p
    return m, {"MAE": float(mean_absolute_error(test.demand_kwh, p)),
               "RMSE": float(np.sqrt(mean_squared_error(test.demand_kwh, p)))}, out


# ----------------------------------------------------------------------- forecast
def future_temperatures(d: pd.DataFrame, horizon: int = HORIZON, mode: str = "repeat_last_day",
                        offset_c: float = 0.0) -> np.ndarray:
    """Temperature assumption for the next ``horizon`` hours, built only from history."""
    if mode not in TEMP_MODES:
        raise ValueError(f"Unknown temperature mode {mode!r}")
    temps = d.temperature_c.to_numpy()
    if mode == "hold_last":
        base = np.full(horizon, temps[-1])
    else:
        last_day = temps[-24:]
        base = np.array([last_day[(k - 1) % 24] for k in range(1, horizon + 1)])
    return base + float(offset_c)


def _recursive(model, histories: list[np.ndarray], last_ts: list[pd.Timestamp],
               temps: list[np.ndarray], horizon: int) -> np.ndarray:
    """Recursive multi-step forecast for several origins at once.

    ``histories[i]`` holds demand up to and including origin i. Predictions are appended
    to a copy of the history, so future actual values are never visible.
    """
    hist = [list(map(float, h)) for h in histories]
    out = np.zeros((len(hist), horizon))
    for k in range(1, horizon + 1):
        rows = []
        for i, v in enumerate(hist):
            t = pd.Timestamp(last_ts[i]) + np.timedelta64(k, "h")
            rows.append({"hour": t.hour, "weekday": t.dayofweek, "temperature_c": temps[i][k - 1],
                         "lag24": v[-24], "lag168": v[-168], "mean24": float(np.mean(v[-24:]))})
        x = pd.DataFrame(rows, columns=FEATURES)
        y = np.maximum(model.predict(x), 0)
        for i, v in enumerate(hist):
            v.append(float(y[i]))
        out[:, k - 1] = y
    return out


def forecast(d: pd.DataFrame, horizon: int = HORIZON, model=None, temp_mode: str = "repeat_last_day",
             temp_offset: float = 0.0) -> pd.DataFrame:
    """Recursive forecast of the next ``horizon`` hours after the last observation."""
    d = validate(d)
    if not 1 <= horizon <= 168:
        raise ValueError("horizon must be between 1 and 168 hours")
    if model is None:
        f = features(d)
        model = make_model().fit(f[FEATURES], f.demand_kwh)
    temps = future_temperatures(d, horizon, temp_mode, temp_offset)
    last = d.timestamp.iloc[-1]
    y = _recursive(model, [d.demand_kwh.to_numpy()], [last], [temps], horizon)[0]
    return pd.DataFrame({"timestamp": [pd.Timestamp(last) + np.timedelta64(k, "h") for k in range(1, horizon + 1)],
                         "forecast_kwh": y, "assumed_temperature_c": temps})


# --------------------------------------------------------------------- evaluation
def _metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    mae = float(mean_absolute_error(y, p))
    return {"MAE": mae, "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAPE": float(np.mean(np.abs((y - p) / np.where(y == 0, np.nan, y))) * 100),
            "Bias": float(np.mean(p - y))}


@dataclass
class Evaluation:
    model: RandomForestRegressor            # refit on all rows, used for the live forecast
    test_model: RandomForestRegressor       # trained on the first 80% only
    one_step: pd.DataFrame                  # model vs baselines on the holdout
    holdout: pd.DataFrame                   # timestamp, demand_kwh, predicted_kwh, baselines
    backtest: pd.DataFrame                  # per-origin per-horizon errors
    backtest_summary: pd.DataFrame
    by_horizon: pd.DataFrame
    intervals: pd.DataFrame                 # empirical error quantiles by horizon
    importance: pd.DataFrame
    meta: dict


def backtest(d: pd.DataFrame, model, start: pd.Timestamp, horizon: int = HORIZON,
             temp_mode: str = "repeat_last_day") -> pd.DataFrame:
    """Day-ahead recursive backtest: one origin per day at 23:00 from ``start`` onward.

    For each origin only data up to the origin is used; temperatures follow ``temp_mode``
    (no future values) and, separately, the actual temperatures (an upper bound).
    """
    d = d.reset_index(drop=True)
    ts = d.timestamp
    origins = d.index[(ts >= start) & (ts.dt.hour == 23)]
    origins = [i for i in origins if i >= 168 and i + horizon < len(d)]
    if not origins:
        return pd.DataFrame()
    y = d.demand_kwh.to_numpy()
    temp = d.temperature_c.to_numpy()
    hists = [y[: i + 1] for i in origins]
    last = [ts.iloc[i] for i in origins]
    assumed = [future_temperatures(d.iloc[: i + 1], horizon, temp_mode) for i in origins]
    actual_t = [temp[i + 1: i + 1 + horizon] for i in origins]
    pred = _recursive(model, hists, last, assumed, horizon)
    pred_oracle = _recursive(model, hists, last, actual_t, horizon)
    rows = []
    for j, i in enumerate(origins):
        for k in range(1, horizon + 1):
            t_idx = i + k
            rows.append({"origin": ts.iloc[i], "timestamp": ts.iloc[t_idx], "horizon": k,
                         "actual": y[t_idx], "forecast": pred[j, k - 1], "forecast_actual_temp": pred_oracle[j, k - 1],
                         "naive_24h": y[t_idx - 24], "naive_168h": y[t_idx - 168]})
    return pd.DataFrame(rows)


def evaluate(d: pd.DataFrame, temp_mode: str = "repeat_last_day") -> Evaluation:
    d = validate(d)
    f = features(d)
    cut = _split(f)
    train, test = f.iloc[:cut], f.iloc[cut:]
    test_model = make_model().fit(train[FEATURES], train.demand_kwh)
    p = test_model.predict(test[FEATURES])

    hold = test[["timestamp", "demand_kwh", "lag24", "lag168"]].copy()
    hold["predicted_kwh"] = p
    one = pd.DataFrame([
        {"Method": "Random Forest (one step ahead)", **_metrics(test.demand_kwh, p)},
        {"Method": "Seasonal naive: same hour yesterday", **_metrics(test.demand_kwh, test.lag24)},
        {"Method": "Seasonal naive: same hour last week", **_metrics(test.demand_kwh, test.lag168)},
    ])

    bt = backtest(d, test_model, test.timestamp.iloc[0], HORIZON, temp_mode)
    if len(bt):
        summ = pd.DataFrame([
            {"Method": "Random Forest, recursive 24h (assumed temperature)", **_metrics(bt.actual, bt.forecast)},
            {"Method": "Random Forest, recursive 24h (actual temperature, upper bound)", **_metrics(bt.actual, bt.forecast_actual_temp)},
            {"Method": "Seasonal naive: same hour yesterday", **_metrics(bt.actual, bt.naive_24h)},
            {"Method": "Seasonal naive: same hour last week", **_metrics(bt.actual, bt.naive_168h)},
        ])
        bt["error"] = bt.forecast - bt.actual
        byh = bt.groupby("horizon").apply(
            lambda g: pd.Series({"MAE": mean_absolute_error(g.actual, g.forecast),
                                 "MAE naive 24h": mean_absolute_error(g.actual, g.naive_24h),
                                 "MAE naive 168h": mean_absolute_error(g.actual, g.naive_168h)}),
            include_groups=False).reset_index()
        ints = bt.groupby("horizon").error.quantile([0.1, 0.9]).unstack().reset_index()
        ints.columns = ["horizon", "q10", "q90"]
    else:
        summ = byh = ints = pd.DataFrame()

    sample = test.sample(min(len(test), 1500), random_state=RANDOM_STATE)
    pi = permutation_importance(test_model, sample[FEATURES], sample.demand_kwh, n_repeats=5,
                                random_state=RANDOM_STATE, scoring="neg_mean_absolute_error", n_jobs=1)
    imp = (pd.DataFrame({"feature": FEATURES, "mae_increase": pi.importances_mean, "std": pi.importances_std})
           .sort_values("mae_increase", ascending=False).reset_index(drop=True))

    full_model = make_model().fit(f[FEATURES], f.demand_kwh)
    meta = {
        "rows": int(len(d)), "feature_rows": int(len(f)),
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "train_period": [str(train.timestamp.iloc[0]), str(train.timestamp.iloc[-1])],
        "test_period": [str(test.timestamp.iloc[0]), str(test.timestamp.iloc[-1])],
        "backtest_origins": int(bt.origin.nunique()) if len(bt) else 0,
        "temp_mode": temp_mode, "random_state": RANDOM_STATE,
        "mean_demand_test": float(test.demand_kwh.mean()),
    }
    return Evaluation(full_model, test_model, one, hold, bt, summ, byh, ints, imp, meta)


# ------------------------------------------------------------------------ anomalies
def anomalies(d: pd.DataFrame, rate: float = 0.03) -> pd.DataFrame:
    """Retrospective anomaly screen with Isolation Forest.

    Each hour is compared with an *expected* value: the median demand for the same
    weekday type and hour within a centred ±14-day window, adjusted for temperature by a
    simple fitted response. The detector uses the residual, the hour-on-hour change and
    the 4-hour rolling spread (to catch a stuck meter). Flags are statistical outliers,
    not confirmed faults.
    """
    if not 0.001 <= rate <= 0.5:
        raise ValueError("rate must be between 0.001 and 0.5")
    d = validate(d).copy()
    v = d.demand_kwh
    hour = d.timestamp.dt.hour
    wtype = (d.timestamp.dt.dayofweek >= 5).astype(int)
    key = wtype * 24 + hour
    expected = pd.Series(np.nan, index=d.index)
    for k, idx in d.groupby(key).groups.items():
        s = v.loc[idx]
        expected.loc[idx] = s.rolling(9, center=True, min_periods=3).median().to_numpy()
    # temperature adjustment from residual vs temperature (piecewise linear fit)
    resid0 = v - expected
    tc = d.temperature_c
    X = np.column_stack([np.ones(len(d)), np.maximum(tc - 22, 0), np.maximum(14 - tc, 0)])
    ok = resid0.notna().to_numpy()
    if ok.sum() > 10:
        coef, *_ = np.linalg.lstsq(X[ok], resid0.to_numpy()[ok], rcond=None)
        expected = expected + X @ coef
    d["expected_kwh"] = expected.bfill().ffill()
    d["residual_kwh"] = v - d["expected_kwh"]
    d["change_kwh"] = v.diff().fillna(0)
    d["spread_4h"] = v.rolling(4, min_periods=2).std().bfill()
    scale = float(np.median(np.abs(d.residual_kwh - d.residual_kwh.median()))) or 1.0
    X_if = np.column_stack([d.residual_kwh / scale, d.change_kwh / scale,
                            np.log1p(d.spread_4h / scale)])
    iso = IsolationForest(n_estimators=200, contamination=rate, random_state=RANDOM_STATE)
    d["anomaly"] = iso.fit_predict(X_if) == -1
    d["anomaly_score"] = -iso.score_samples(X_if)
    d["kind"] = np.where(~d.anomaly, "",
                         np.where(d.spread_4h < 0.05 * scale, "Flat / stuck reading",
                                  np.where(d.residual_kwh > 0, "Higher than expected", "Lower than expected")))
    return d


def anomaly_scores(flags: pd.DataFrame, labels: pd.Series) -> dict:
    """Precision/recall of the screen against known (injected) synthetic labels."""
    y = np.asarray(labels, bool)
    p = flags.anomaly.to_numpy(bool)
    tp = int((y & p).sum())
    fp = int((~y & p).sum())
    fn = int((y & ~p).sum())
    return {"precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "labelled": int(y.sum()), "flagged": int(p.sum())}


# ----------------------------------------------------------------------------- cost
@dataclass
class Tariff:
    currency: str = "USD"
    mode: str = "time_of_use"         # "flat" or "time_of_use"
    flat_rate: float = 0.18
    peak_rate: float = 0.26
    offpeak_rate: float = 0.12
    peak_start: int = 8
    peak_end: int = 20                # exclusive
    weekend_offpeak: bool = True
    demand_charge: float = 0.0        # per kW of monthly peak hourly demand
    fixed_monthly: float = 0.0

    def validate(self) -> "Tariff":
        if self.mode not in ("flat", "time_of_use"):
            raise ValueError("mode must be 'flat' or 'time_of_use'")
        for name in ("flat_rate", "peak_rate", "offpeak_rate", "demand_charge", "fixed_monthly"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if not (0 <= self.peak_start <= 23 and 1 <= self.peak_end <= 24 and self.peak_start < self.peak_end):
            raise ValueError("peak window must satisfy 0 <= start < end <= 24")
        return self


def hourly_rates(ts: pd.Series, tariff: Tariff) -> np.ndarray:
    tariff.validate()
    ts = pd.to_datetime(ts)
    if tariff.mode == "flat":
        return np.full(len(ts), tariff.flat_rate)
    h = ts.dt.hour.to_numpy()
    peak = (h >= tariff.peak_start) & (h < tariff.peak_end)
    if tariff.weekend_offpeak:
        peak &= ts.dt.dayofweek.to_numpy() < 5
    return np.where(peak, tariff.peak_rate, tariff.offpeak_rate)


def energy_cost(df: pd.DataFrame, tariff: Tariff, kwh_col: str = "demand_kwh") -> pd.DataFrame:
    """Hourly cost table: energy cost per hour plus the peak flag."""
    out = df[["timestamp", kwh_col]].copy()
    out["rate"] = hourly_rates(out.timestamp, tariff)
    out["period"] = np.where(out.rate == tariff.peak_rate, "Peak", "Off-peak") if tariff.mode == "time_of_use" else "Flat"
    out["energy_cost"] = out[kwh_col] * out.rate
    return out


def monthly_bill(df: pd.DataFrame, tariff: Tariff) -> pd.DataFrame:
    """Illustrative monthly bill: energy + demand charge (monthly peak kW) + fixed."""
    c = energy_cost(df, tariff)
    c["month"] = c.timestamp.dt.to_period("M").astype(str)
    g = c.groupby("month").agg(kwh=("demand_kwh", "sum"), energy_cost=("energy_cost", "sum"),
                               peak_kw=("demand_kwh", "max"), hours=("demand_kwh", "size")).reset_index()
    # hourly kWh equals average kW over that hour
    g["demand_charge"] = g.peak_kw * tariff.demand_charge
    g["fixed"] = tariff.fixed_monthly
    g["total"] = g.energy_cost + g.demand_charge + g.fixed
    g["partial_month"] = g.hours < g.month.map(lambda m: pd.Period(m).days_in_month * 24)
    return g


# ---------------------------------------------------------------------- persistence
def save_model(ev: Evaluation, path: str | Path) -> Path:
    import joblib
    import sklearn
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rf = ev.one_step.iloc[0]
    joblib.dump({"model": ev.model, "features": FEATURES, "sklearn_version": sklearn.__version__,
                 "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "holdout_one_step": {"MAE": float(rf.MAE), "RMSE": float(rf.RMSE)},
                 "meta": ev.meta}, path)
    return path


def load_model(path: str | Path) -> dict:
    import warnings

    import joblib
    import sklearn
    b = joblib.load(path)
    if b.get("features") != FEATURES:
        raise ValueError("Saved model was trained with different features")
    if b.get("sklearn_version") != sklearn.__version__:
        warnings.warn(f"Model saved with scikit-learn {b.get('sklearn_version')}, running {sklearn.__version__}")
    return b


def evaluation_json(ev: Evaluation) -> str:
    return json.dumps({
        "meta": ev.meta,
        "one_step_holdout": ev.one_step.round(4).to_dict(orient="records"),
        "recursive_backtest": ev.backtest_summary.round(4).to_dict(orient="records"),
        "permutation_importance": ev.importance.round(4).to_dict(orient="records"),
    }, indent=2)
