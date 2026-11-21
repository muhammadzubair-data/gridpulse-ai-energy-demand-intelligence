"""Browser end-to-end check of the running GridPulse dashboard (Playwright + Chromium).

Not collected by pytest (file name). Usage:
    streamlit run app.py --server.port 8501 &
    python tests/e2e_ui.py http://localhost:8501 [--shots DIR] [--chrome /path/to/chrome]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

PAGES = [("", "command"), ("forecast", "forecast"), ("anomalies", "anomalies"), ("cost", "cost"), ("data", "data"), ("model", "model")]
MIN_CHARTS = {"command": 5, "forecast": 2, "anomalies": 3, "cost": 2, "data": 2, "model": 6}


def idle(page, extra=2.0):
    page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=180_000)
    time.sleep(1.5)
    for _ in range(600):
        if page.locator("[data-testid='stSpinner']").count() + page.locator("[data-testid='stStatusWidget']").count() == 0:
            break
        time.sleep(0.5)
    time.sleep(extra)


def check(page, name, failures, min_charts=0):
    exc = page.locator("[data-testid='stException']").count()
    errs = [e.inner_text()[:160] for e in page.locator("[data-testid='stAlertContentError']").all()]
    charts = page.locator(".js-plotly-plot").count()
    print(f"  {name:<14} exceptions={exc} errors={len(errs)} charts={charts}")
    if exc or errs or charts < min_charts:
        failures.append((name, exc, errs, charts))


def kpi(page, i):
    return page.locator("[data-testid='stMain'] .kpi .value").nth(i).inner_text()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?", default="http://localhost:8501")
    ap.add_argument("--shots")
    ap.add_argument("--chrome", default=os.environ.get("CHROME_PATH"))
    a = ap.parse_args()
    base = a.url.rstrip("/")
    failures: list = []
    with sync_playwright() as p:
        launch = {"args": ["--no-sandbox"]}
        if a.chrome:
            launch["executable_path"] = a.chrome
        browser = p.chromium.launch(**launch)
        for theme in (["light", "dark"] if a.shots else ["light"]):
            print(f"Pages ({theme}):")
            page = browser.new_page(viewport={"width": 1440, "height": 900}, color_scheme=theme)
            for path, name in PAGES:
                page.goto(f"{base}/{path}", wait_until="networkidle")
                idle(page)
                check(page, name, failures, MIN_CHARTS.get(name, 0))
                if a.shots:
                    Path(a.shots).mkdir(parents=True, exist_ok=True)
                    page.screenshot(path=f"{a.shots}/{name}-{theme}.png")
            page.close()

        print("Interactions:")
        page = browser.new_page(viewport={"width": 1440, "height": 900}, accept_downloads=True)
        page.goto(f"{base}/forecast", wait_until="networkidle")
        idle(page)
        before = kpi(page, 0)
        page.get_by_role("group", name="Temperature adjustment (°C)").get_by_role("slider").focus()
        for _ in range(8):
            page.keyboard.press("ArrowRight")
        idle(page, 2)
        after = kpi(page, 0)
        print(f"  forecast       +4 °C temperature scenario changed forecast energy {before} -> {after}")
        if before == after:
            failures.append(("forecast", "scenario did not change"))
        check(page, "forecast-edit", failures)

        page.goto(f"{base}/cost", wait_until="networkidle")
        idle(page)
        before = kpi(page, 0)
        page.get_by_text("Flat rate").click()
        idle(page, 2)
        after = kpi(page, 0)
        print(f"  cost           tariff switch changed energy cost {before} -> {after}")
        if before == after:
            failures.append(("cost", "tariff did not change cost"))

        print("Downloads:")
        downloads = [("forecast", "Download forecast CSV"), ("anomalies", "Download flagged hours CSV"),
                     ("cost", "Download monthly cost CSV"), ("data", "Download readings in view (CSV)"),
                     ("data", "Download full dataset (CSV)"), ("data", "Download evaluation (JSON)")]
        for path, label in downloads:
            page.goto(f"{base}/{path}", wait_until="networkidle")
            idle(page)
            with page.expect_download() as d:
                page.get_by_role("button", name=label).click()
            f = Path(d.value.path())
            size = f.stat().st_size
            print(f"  {d.value.suggested_filename:<34} {size:,} bytes")
            if size < 200:
                failures.append(("download", label, size))
            if d.value.suggested_filename.endswith(".json"):
                assert "recursive_backtest" in json.loads(f.read_text())

        print("Upload:")
        page.goto(f"{base}/data", wait_until="networkidle")
        idle(page)
        tmp = Path(tempfile.mkdtemp())
        bad = tmp / "bad.csv"
        bad.write_text("timestamp,demand_kwh\n2025-01-01 00:00,10\n")
        page.locator("input[type='file']").set_input_files(str(bad))
        idle(page, 2)
        msg = page.locator("[data-testid='stAlertContentError']").first.inner_text()
        print(f"  invalid file   rejected: {msg[:90]!r}")
        if "temperature_c" not in msg:
            failures.append(("upload", "invalid file not explained"))
        with page.expect_download() as d:
            page.get_by_role("button", name="Download a template CSV").click()
        good = tmp / "my_site.csv"
        Path(d.value.path()).replace(good)
        page.locator("input[type='file']").set_input_files(str(good))
        idle(page, 2)
        page.get_by_role("button", name="Analyse this file").click()
        idle(page, 4)
        ok = page.get_by_text("Uploaded data: my_site.csv").count() > 0
        print(f"  valid file     analysed as uploaded data: {ok}")
        if not ok:
            failures.append(("upload", "upload not activated"))
        # move between pages with the sidebar (same session): the upload must stay the active data source
        titles = {"command": "Energy Command Center", "forecast": "Demand Forecasting", "anomalies": "Anomaly Intelligence",
                  "cost": "Cost Intelligence", "data": "Data Explorer", "model": "Model Performance"}
        for path, name in PAGES:
            page.get_by_role("link", name=titles[name]).click()
            idle(page, 2)
            kept = page.get_by_text("Uploaded data: my_site.csv").count() > 0
            if not kept:
                failures.append((f"upload-{name}", "data source reset after page switch"))
            check(page, f"upload-{name}", failures, 1)
        page.get_by_text("Synthetic demo", exact=True).click()
        idle(page, 3)
        back = page.get_by_text("Synthetic demonstration data").count() > 0
        page.get_by_role("link", name="Demand Forecasting").click()
        idle(page, 2)
        still = page.get_by_text("Synthetic demonstration data").count() > 0
        print(f"  switch back    demo restored: {back}, kept after page switch: {still}")
        if not (back and still):
            failures.append(("upload", "could not switch back to synthetic demo"))
        browser.close()
    if failures:
        print("FAILED:", failures)
        return 1
    print("E2E OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
