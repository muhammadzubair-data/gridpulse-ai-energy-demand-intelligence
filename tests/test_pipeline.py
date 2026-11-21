"""Pipeline, leakage, validation, cost, persistence and app tests for GridPulse AI."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridpulse import core
from gridpulse.core import (FEATURES, Tariff, anomalies, anomaly_scores, backtest, demo, energy_cost, evaluate,
                            features, fit, forecast, future_temperatures, hourly_rates, load_model, monthly_bill,
                            save_model, validate, validate_report)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def year():
    return demo(365, 42)


@pytest.fixture(scope="module")
def ev120():
    return evaluate(demo(120, 42))


# ------------------------------------------------------------------ data
def test_demo_is_realistic(year):
    assert list(year.columns) == ["timestamp", "demand_kwh", "temperature_c", "injected_anomaly"]
    assert year.timestamp.diff().dropna().eq(pd.Timedelta(hours=1)).all()
    assert (year.demand_kwh >= 0).all()
    summer = year[year.timestamp.dt.month == 7].temperature_c.mean()
    winter = year[year.timestamp.dt.month == 1].temperature_c.mean()
    assert summer > winter + 10
    clean = year[~year.injected_anomaly]
    weekday = clean[clean.timestamp.dt.dayofweek < 5].demand_kwh.mean()
    weekend = clean[clean.timestamp.dt.dayofweek >= 5].demand_kwh.mean()
    assert weekday > weekend
    assert 0.005 < year.injected_anomaly.mean() < 0.05


def test_demo_seed_and_flags():
    assert demo(30, 1).equals(demo(30, 1))
    assert not demo(30, 1).equals(demo(30, 2))
    assert not demo(30, anomalies=False).injected_anomaly.any()
    with pytest.raises(ValueError):
        demo(0)


def test_sample_csv_matches_generator():
    sample = pd.read_csv(ROOT / "sample_data" / "demo_hourly.csv")
    clean = validate(sample)
    ref = demo(60, 42)
    assert len(clean) == len(ref)
    assert np.allclose(clean.demand_kwh, ref.demand_kwh, atol=1e-3)


# ------------------------------------------------------------------ validation
@pytest.mark.parametrize("mutate, fragment", [
    (lambda d: d.drop(columns="temperature_c"), "temperature_c"),
    (lambda d: d.drop(index=[200, 201]), "continuous"),
    (lambda d: d.assign(demand_kwh=d.demand_kwh.astype(object).where(d.index != 5, "abc")), "non-numeric"),
    (lambda d: d.head(100), "15+ days"),
    (lambda d: d.assign(temperature_c=d.temperature_c.where(d.index != 3, 99.0)), "between -60 and 60"),
])
def test_validate_rejects(mutate, fragment):
    d = demo(20)[["timestamp", "demand_kwh", "temperature_c"]]
    with pytest.raises(ValueError, match=re.escape(fragment)):
        validate(mutate(d))


def test_validate_accepts_messy_headers_and_timezones():
    d = demo(20)[["timestamp", "demand_kwh", "temperature_c"]]
    d.columns = [" Timestamp", "DEMAND_KWH ", "temperature_c"]
    d[" Timestamp"] = pd.to_datetime(d[" Timestamp"]).dt.tz_localize("UTC").astype(str)
    out = validate(d.sample(frac=1, random_state=0))
    assert out.timestamp.is_monotonic_increasing and out.timestamp.dt.tz is None


def test_validate_report_repairs_small_issues():
    d = demo(20)[["timestamp", "demand_kwh", "temperature_c"]]
    messy = pd.concat([d.drop(index=[100, 101]), d.iloc[[50]]])
    clean, rep = validate_report(messy, repair=False)
    assert clean is None and any("gap" in p for p in rep.problems) and any("duplicate" in p for p in rep.problems)
    clean, rep = validate_report(messy, repair=True)
    assert clean is not None and rep.ok
    assert rep.duplicates_merged == 1 and rep.gaps_filled == 2 and len(clean) == len(d)
    long_gap = d.drop(index=range(100, 106))
    clean, rep = validate_report(long_gap, repair=True)
    assert clean is None and rep.problems


# ------------------------------------------------------------------ features and leakage
def test_features_use_only_past_values():
    d = demo(20)
    f0 = features(d)
    changed = d.copy()
    i = 300
    changed.loc[i, "demand_kwh"] += 1000
    f1 = features(changed)
    row = f0.index[f0.timestamp == d.timestamp[i]][0]
    pd.testing.assert_frame_equal(f0.loc[:row, FEATURES], f1.loc[:row, FEATURES])
    assert not f0.loc[row + 1:, FEATURES].equals(f1.loc[row + 1:, FEATURES])
    assert set(FEATURES).isdisjoint({"demand_kwh", "timestamp"})


def test_recursive_feeds_back_its_own_predictions():
    class EchoMean24:
        def predict(self, x):
            return x["mean24"].to_numpy()
    hist = np.arange(1.0, 201.0)
    out = core._recursive(EchoMean24(), [hist], [pd.Timestamp("2025-01-10 23:00")], [np.zeros(3)], 3)[0]
    first = hist[-24:].mean()
    second = np.mean(list(hist[-23:]) + [first])
    assert out[0] == pytest.approx(first) and out[1] == pytest.approx(second)


def test_forecast_never_sees_future_actuals(year):
    model = core.make_model(30).fit(features(year)[FEATURES], features(year).demand_kwh)
    a = backtest(year, model, year.timestamp.iloc[4000], horizon=24)
    tampered = year.copy()
    tampered.loc[4500:, "demand_kwh"] *= 3
    b = backtest(tampered, model, year.timestamp.iloc[4000], horizon=24)
    origin = year.timestamp.iloc[4500 - 1]
    first_affected = a[a.origin >= origin].origin.min()
    keep = a.origin < first_affected
    np.testing.assert_allclose(a[keep].forecast, b[keep].forecast)
    # origins after the change see tampered history, so their forecasts may differ
    assert not np.allclose(a[~keep].forecast, b[~keep].forecast)


def test_forecast_output_and_temperature_assumptions(year):
    d = year[["timestamp", "demand_kwh", "temperature_c"]]
    fc = forecast(d.tail(24 * 30))
    assert len(fc) == 24 and (fc.forecast_kwh >= 0).all()
    assert fc.timestamp.iloc[0] == d.timestamp.iloc[-1] + pd.Timedelta(hours=1)
    np.testing.assert_allclose(fc.assumed_temperature_c, d.temperature_c.tail(24).to_numpy())
    hold = future_temperatures(d, 5, "hold_last", 2.0)
    assert np.allclose(hold, d.temperature_c.iloc[-1] + 2.0)
    with pytest.raises(ValueError):
        future_temperatures(d, 5, "magic")
    with pytest.raises(ValueError):
        forecast(d.tail(24 * 30), horizon=500)


# ------------------------------------------------------------------ evaluation
def test_evaluation_is_chronological_and_complete(ev120):
    m = ev120.meta
    assert pd.Timestamp(m["train_period"][1]) < pd.Timestamp(m["test_period"][0])
    assert ev120.backtest.origin.min() >= pd.Timestamp(m["test_period"][0])
    assert set(ev120.backtest.horizon.unique()) == set(range(1, 25))
    assert list(ev120.one_step.columns) == ["Method", "MAE", "RMSE", "MAPE", "Bias"]
    assert len(ev120.backtest_summary) == 4 and (ev120.intervals.q10 < ev120.intervals.q90).all()
    assert set(ev120.importance.feature) == set(FEATURES)


def test_model_beats_seasonal_naive_on_synthetic_data(ev120):
    bt = ev120.backtest_summary.set_index("Method").MAE
    rf = bt.iloc[0]
    assert rf < bt.iloc[2:].min()
    one = ev120.one_step.set_index("Method").MAE
    assert one.iloc[0] < one.iloc[1:].min()


def test_fit_is_backward_compatible_and_reproducible():
    d = demo(40)
    _, m1, out1 = fit(d)
    _, m2, out2 = fit(d)
    assert m1 == pytest.approx(m2) and {"MAE", "RMSE"} <= set(m1)
    assert {"timestamp", "demand_kwh", "predicted_kwh"} <= set(out1.columns)


# ------------------------------------------------------------------ anomalies
def test_anomaly_screen_finds_injected_events(year):
    flags = anomalies(year[["timestamp", "demand_kwh", "temperature_c"]], 0.02)
    assert {"expected_kwh", "residual_kwh", "anomaly", "kind", "anomaly_score"} <= set(flags.columns)
    assert abs(flags.anomaly.mean() - 0.02) < 0.005
    sc = anomaly_scores(flags, year.injected_anomaly)
    assert sc["recall"] > 0.5 and sc["precision"] > 0.4
    with pytest.raises(ValueError):
        anomalies(year, 0.9)


# ------------------------------------------------------------------ cost
def test_tariffs_and_bills():
    ts = pd.Series(pd.date_range("2025-01-06", periods=48, freq="h"))  # Monday
    df = pd.DataFrame({"timestamp": ts, "demand_kwh": 10.0})
    flat = energy_cost(df, Tariff(mode="flat", flat_rate=0.2))
    assert flat.energy_cost.sum() == pytest.approx(48 * 10 * 0.2)
    tou = Tariff(peak_rate=0.3, offpeak_rate=0.1, peak_start=8, peak_end=20)
    rates = hourly_rates(ts, tou)
    assert rates[8] == 0.3 and rates[7] == 0.1 and rates[20] == 0.1
    weekend = pd.Series(pd.date_range("2025-01-11 12:00", periods=1, freq="h"))  # Saturday noon
    assert hourly_rates(weekend, tou)[0] == 0.1
    assert hourly_rates(weekend, Tariff(weekend_offpeak=False))[0] == Tariff().peak_rate
    bill = monthly_bill(df.assign(demand_kwh=np.r_[np.full(47, 10.0), 25.0]), Tariff(mode="flat", flat_rate=0.2,
                                                                                    demand_charge=5, fixed_monthly=30))
    row = bill.iloc[0]
    assert row.peak_kw == 25 and row.demand_charge == 125 and row.total == pytest.approx(row.energy_cost + 125 + 30)
    assert bool(row.partial_month)
    for bad in (Tariff(mode="x"), Tariff(peak_rate=-1), Tariff(peak_start=20, peak_end=8)):
        with pytest.raises(ValueError):
            bad.validate()


# ------------------------------------------------------------------ persistence and CLI
def test_save_load_roundtrip(tmp_path, ev120):
    p = save_model(ev120, tmp_path / "m.joblib")
    b = load_model(p)
    x = features(demo(20))[FEATURES].head(5)
    np.testing.assert_allclose(b["model"].predict(x), ev120.model.predict(x))
    b["features"] = ["x"]
    import joblib
    joblib.dump(b, tmp_path / "bad.joblib")
    with pytest.raises(ValueError):
        load_model(tmp_path / "bad.joblib")


def test_train_script(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("train", ROOT / "scripts" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    summary = mod.main(["--days", "60", "--out", str(tmp_path)])
    assert (tmp_path / "gridpulse_model.joblib").exists() and (tmp_path / "metrics.json").exists()
    assert summary["recursive_backtest"][0]["MAE"] > 0
    summary = mod.main(["--csv", str(ROOT / "sample_data" / "demo_hourly.csv"), "--out", str(tmp_path / "csv")])
    assert summary["meta"]["rows"] == 60 * 24


# ------------------------------------------------------------------ app
TITLES = {"Energy Command Center": "Energy Command Center", "forecast": "Demand Forecasting", "anomalies": "Anomaly Intelligence",
          "cost": "Cost Intelligence", "data": "Data Explorer", "model": "Model Performance"}


@pytest.mark.parametrize("page", list(TITLES))
def test_app_pages_render(page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240)
    at.run()
    if page != "Energy Command Center":
        # pages are callables registered by st.navigation, so select them by url path
        target = [h for h, info in at._registered_pages.items() if info.get("url_pathname") == page]
        assert target, f"page {page!r} not registered"
        at._page_hash = target[0]
        at.run()
    assert any(f"<div class='sec-title'>{TITLES[page]}</div>" in m.value for m in at.markdown), "page did not switch"
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
