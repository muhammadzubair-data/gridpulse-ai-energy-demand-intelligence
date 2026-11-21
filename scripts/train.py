"""Train, evaluate and save the GridPulse forecasting model.

    python scripts/train.py                       # 365 days of synthetic data, seed 42
    python scripts/train.py --csv my_meter.csv    # your own hourly data
    python scripts/train.py --days 730 --seed 7 --out artifacts
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridpulse.core import demo, evaluate, evaluation_json, save_model, validate_report  # noqa: E402


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", help="hourly CSV with timestamp, demand_kwh, temperature_c")
    ap.add_argument("--repair", action="store_true", help="merge duplicates and fill gaps up to 3 hours")
    ap.add_argument("--out", default="artifacts")
    a = ap.parse_args(argv)
    if a.csv:
        data, rep = validate_report(pd.read_csv(a.csv), repair=a.repair)
        if data is None:
            raise SystemExit("Invalid data: " + "; ".join(rep.problems))
    else:
        data = demo(a.days, a.seed)
    ev = evaluate(data)
    out = Path(a.out)
    save_model(ev, out / "gridpulse_model.joblib")
    (out / "metrics.json").write_text(evaluation_json(ev))
    summary = json.loads(evaluation_json(ev))
    bt = summary["recursive_backtest"][0] if summary["recursive_backtest"] else {}
    print(f"Day-ahead recursive MAE {bt.get('MAE', float('nan')):.2f} kWh · one-step MAE "
          f"{summary['one_step_holdout'][0]['MAE']:.2f} kWh -> saved to {out}")
    return summary


if __name__ == "__main__":
    main()
