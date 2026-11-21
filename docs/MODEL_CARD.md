# Model card: GridPulse AI demand forecaster and anomaly screen

## Summary

| Item | Value |
|:--|:--|
| Task | Forecast hourly electricity demand (kWh) for the next 24 hours; flag unusual hours in history |
| Forecast model | `RandomForestRegressor(n_estimators=120, max_depth=14, min_samples_leaf=3, random_state=42)` |
| Forecast method | Recursive: each predicted hour feeds the 24-hour mean used by the next |
| Anomaly model | `IsolationForest(n_estimators=200, contamination=rate, random_state=42)` on residual, hourly change and 4-hour spread |
| Default data | 8,760 synthetic hourly readings (365 days, seed 42) |
| Evaluation | Chronological 80/20 split; one-step holdout; day-ahead recursive backtest on 71 test days |
| Owner | Muhammad Zubair, ZubairLabs |
| Status | Educational portfolio prototype. Not for grid operations, billing or trading. |

## Intended use

Showing how a short-term load forecast should be built and judged: past-only features, an explicit assumption for unknown future inputs, a backtest that mirrors real use, and comparison with simple baselines. Useful for learning, portfolio review, and as a starting template for a site's own hourly data.

**Not intended for:** dispatch, procurement or trading decisions, billing, or any claim about real-world forecast accuracy or savings.

## Inputs

| Feature | Known at forecast time |
|:--|:--|
| `hour`, `weekday` | Yes (calendar) |
| `lag24`, `lag168` | Yes for horizons up to 24 h (observed demand) |
| `mean24` | First step observed; later steps include earlier predictions |
| `temperature_c` | No. History uses measured values; forecasts use an assumption (default: repeat the last 24 observed hours; option: hold the last value; optional ± offset) |

## Training data

Simulated commercial site: weekday business-hours profile, weekend profile, heating below 14 °C, cooling above 22 °C, 3% yearly trend, AR(1) noise, annual, daily and multi-day temperature variation, plus 60 injected anomaly events (167 hours) labelled for scoring. No holidays.

## Results (365 days, seed 42)

Day-ahead recursive backtest, 71 origins × 24 hours:

| Method | MAE | RMSE | MAPE | Bias |
|:--|--:|--:|--:|--:|
| Random Forest, assumed temperature | 6.55 | 10.23 | 5.6% | −2.30 |
| Random Forest, actual temperature (upper bound) | 5.77 | 9.54 | 4.9% | −2.03 |
| Same hour yesterday | 10.29 | 16.09 | 8.7% | −0.23 |
| Same hour last week | 8.37 | 13.41 | 7.2% | −1.15 |

One-step holdout (1,719 hours): MAE 5.74, RMSE 9.46, MAPE 4.9%.

Permutation importance (holdout MAE increase, kWh): lag168 2.82, temperature 2.40, hour 2.30, weekday 1.33, lag24 1.12, mean24 0.39.

Empirical 80% band at horizon 1: −11.0 to +8.0 kWh around the forecast; at horizon 24: −11.2 to +7.4 kWh.

Anomaly screen at 2%: 176 hours flagged; precision 0.65, recall 0.68 (114 of 167 injected hours) with the pinned dependencies used in CI and on Streamlit Cloud (0.64 with pandas 3).

## Leakage safeguards (tested)

- Lag and rolling features use `shift`, so a reading only affects later rows.
- Recursive forecasts append predictions to a copy of history, never actual future values.
- Backtest forecasts made before a change in the data stay identical when later actuals are altered.
- The backtest model is trained only on the training period; the live model is refit on all rows.

## Limitations

- Results are on simulated data and do not transfer to real sites.
- Temperature is a persistence assumption, not a weather forecast.
- One site, one test period, no retraining schedule or drift monitoring.
- The error band is empirical, not calibrated; horizons beyond 24 h are not backtested.
- The anomaly screen uses neighbouring weeks on both sides (retrospective), and flags are not confirmed faults.

## Reproduce

```bash
python scripts/train.py --days 365 --seed 42 --out artifacts
```
