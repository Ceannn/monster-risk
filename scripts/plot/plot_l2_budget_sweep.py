#!/usr/bin/env python3
"""Plot the L2 budget sweep CSV produced by risk-bench.

Usage:
  python3 scripts/plot/plot_l2_budget_sweep.py results/l2_budget_sweep_YYYYMMDD_HHMMSS/sweep_l2_budget.csv

It will emit 4 PNG files alongside the CSV:
  - p99_us_vs_l2_pct.png
  - ok_rps_vs_l2_pct.png
  - http_429_vs_l2_pct.png
  - queue_wait_p99_us_vs_l2_pct.png
"""

import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> int:
    if len(sys.argv) < 2:
        print("need csv path")
        return 2

    csv_path = Path(sys.argv[1])
    out_dir = csv_path.parent

    rows = []
    with csv_path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            label = row.get("label", "")
            m = re.search(r"l2_pct=(\d+)", label)
            if not m:
                # fallback: try parse from l2_budget
                pct = None
            else:
                pct = int(m.group(1))

            def to_float(k: str):
                v = row.get(k, "")
                try:
                    return float(v) if v != "" else None
                except Exception:
                    return None

            rows.append(
                {
                    "l2_pct": pct,
                    "p99_us": to_float("p99_us"),
                    "ok_rps": to_float("ok_rps"),
                    "http_429": to_float("http_429"),
                    "queue_wait_p99_us": to_float("xgb_pool_queue_wait_p99_us"),
                }
            )

    rows = [x for x in rows if x["l2_pct"] is not None]
    rows.sort(key=lambda x: x["l2_pct"])

    x = [r["l2_pct"] for r in rows]

    def plot(y_key: str, title: str, fname: str, y_label: str):
        y = [r[y_key] for r in rows]
        plt.figure()
        plt.plot(x, y, marker="o")
        plt.xlabel("L2 trigger rate (%)")
        plt.ylabel(y_label)
        plt.title(title)
        plt.xticks(x)
        plt.grid(True, linewidth=0.3)
        out = out_dir / fname
        plt.tight_layout()
        plt.savefig(out)
        print(f"wrote {out}")

    plot("p99_us", "P99 latency vs L2 trigger rate", "p99_us_vs_l2_pct.png", "p99 (us)")
    plot("ok_rps", "OK RPS vs L2 trigger rate", "ok_rps_vs_l2_pct.png", "ok_rps")
    plot("http_429", "HTTP 429 count vs L2 trigger rate", "http_429_vs_l2_pct.png", "http_429")
    plot(
        "queue_wait_p99_us",
        "xgb_pool queue_wait p99 vs L2 trigger rate",
        "queue_wait_p99_us_vs_l2_pct.png",
        "queue_wait_p99 (us)",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
