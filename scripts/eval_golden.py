#!/usr/bin/env python3
"""
Golden-set evaluation for /analyze.

Runs every image URL in data/golden_set.json through the API and writes a CSV
with extracted fields, fraud score, latency. Run once with FLOW_VERSION=1 and
once with FLOW_VERSION=2 on the server, then compare the two CSVs.

golden_set.json format:
[
  {"imageUrl": "https://...", "label": "genuine",       "expected_total": 57.50},
  {"imageUrl": "https://...", "label": "screen_photo"},
  {"imageUrl": "https://...", "label": "edited"}
]

Usage:
  python scripts/eval_golden.py --base-url http://localhost:8000 --api-key <PROJECT_KEY> --out results_v2.csv
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import requests


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--golden", default="data/golden_set.json")
    ap.add_argument("--out", default="eval_results.csv")
    args = ap.parse_args()

    golden_path = Path(args.golden)
    if not golden_path.exists():
        print(f"Golden set not found: {golden_path}")
        print('Create it, e.g.: [{"imageUrl": "https://...", "label": "genuine"}]')
        return 1

    cases = json.loads(golden_path.read_text(encoding="utf-8"))
    rows = []

    for i, case in enumerate(cases, 1):
        url = case["imageUrl"]
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{args.base_url}/analyze",
                headers={"X-API-Key": args.api_key},
                data={
                    "userReference": "golden-eval",
                    "scanReference": f"golden-{i}",
                    "imageUrl": url,
                },
                timeout=120,
            )
            elapsed = time.perf_counter() - t0
            data = resp.json().get("data", {})
        except Exception as e:
            rows.append({"case": i, "label": case.get("label"), "error": str(e)})
            print(f"[{i}/{len(cases)}] ERROR: {e}")
            continue

        expected_total = case.get("expected_total")
        total_ok = (
            None
            if expected_total is None
            else (data.get("Total") is not None
                  and abs(float(data["Total"]) - float(expected_total)) <= 0.01)
        )
        # A "genuine" receipt should score low; anything else should score high
        label = case.get("label", "genuine")
        fraud = data.get("fraudScore")
        verdict_ok = None
        if fraud is not None:
            verdict_ok = (fraud < 50) if label == "genuine" else (fraud >= 50)

        rows.append({
            "case": i,
            "label": label,
            "latency_s": round(elapsed, 2),
            "MerchantName": data.get("MerchantName"),
            "MerchantId": data.get("MerchantId"),
            "Total": data.get("Total"),
            "expected_total": expected_total,
            "total_ok": total_ok,
            "fraudScore": fraud,
            "verdict_ok": verdict_ok,
            "confidentScore": data.get("confidentScore"),
            "qrVerified": data.get("qrVerified"),
            "arbitrated": data.get("arbitrated"),
            "reason": (data.get("reason") or "")[:120],
            "error": "",
        })
        print(f"[{i}/{len(cases)}] {label:<14} fraud={fraud} total={data.get('Total')} "
              f"({elapsed:.1f}s)")

    fieldnames = list({k for r in rows for k in r})
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    scored = [r for r in rows if r.get("verdict_ok") is not None]
    if scored:
        acc = sum(1 for r in scored if r["verdict_ok"]) / len(scored)
        print(f"\nVerdict accuracy: {acc:.0%} ({len(scored)} scored cases)")
    lat = [r["latency_s"] for r in rows if r.get("latency_s")]
    if lat:
        print(f"Avg latency: {sum(lat)/len(lat):.1f}s")
    print(f"Results written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
