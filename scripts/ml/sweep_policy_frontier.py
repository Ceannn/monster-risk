#!/usr/bin/env python3
"""Sweep a scalar score/uncertainty threshold and print the frontier.

Use cases:
- score frontier: score=prob, flagged=score>=th
- uncertainty frontier: score=tree_std, flagged=score>=th (route uncertain)

This is intentionally minimal and works with `.f32` scalar files.

Example:
  python3 scripts/ml/sweep_scalar_frontier.py \
    --scalar-f32le results/l1_tree_std_20k.f32 \
    --labels-u8 results/l1_labels_20k.u8 \
    --fracs 0.02,0.05,0.10,0.20

It will choose the threshold that flags approximately each fraction
of the population (top-frac by scalar).
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np


def die(msg: str) -> "NoReturn":
    raise SystemExit(msg)


def parse_fracs(s: str) -> List[float]:
    out: List[float] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        die("--fracs is empty")
    for f in out:
        if not (0.0 < f < 1.0):
            die(f"bad frac: {f}")
    return out


def threshold_for_top_frac_desc(x: np.ndarray, frac: float) -> float:
    n = int(x.size)
    k = int(np.ceil(frac * n))
    k = max(1, min(n, k))
    # kth largest => partition at n-k
    th = float(np.partition(x, n - k)[n - k])
    return th


def pr_stats(
    flagged: np.ndarray, y: np.ndarray
) -> Tuple[int, int, int, int, float, float]:
    tp = int(((flagged == 1) & (y == 1)).sum())
    fp = int(((flagged == 1) & (y == 0)).sum())
    fn = int(((flagged == 0) & (y == 1)).sum())
    tn = int(((flagged == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return tp, fp, fn, tn, rec, prec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scalar-f32le", required=True)
    ap.add_argument("--labels-u8", required=True)
    ap.add_argument(
        "--fracs",
        default="0.02,0.05,0.10,0.15,0.20,0.30,0.50",
        help="Comma-separated fractions to flag.",
    )
    ap.add_argument(
        "--invert",
        action="store_true",
        help="If set, flag the *bottom* frac instead of top frac.",
    )
    args = ap.parse_args()

    x = np.fromfile(args.scalar_f32le, dtype="<f4")
    y = np.fromfile(args.labels_u8, dtype=np.uint8)
    if x.size != y.size:
        die(f"size mismatch: scalar={x.size} labels={y.size}")

    fracs = parse_fracs(args.fracs)

    pos = int((y == 1).sum())
    print(
        f"[SWEEP] n={x.size} pos={pos} pos_rate={pos / max(1, x.size):.6f} invert={args.invert}"
    )

    for frac in fracs:
        if not args.invert:
            th = threshold_for_top_frac_desc(x, frac)
            flagged = (x >= th).astype(np.uint8)
        else:
            # bottom frac => threshold on -x top frac
            th = -threshold_for_top_frac_desc(-x, frac)
            flagged = (x <= th).astype(np.uint8)

        tp, fp, fn, tn, rec, prec = pr_stats(flagged, y)
        flagged_frac = float(flagged.mean())
        print(
            f"[SWEEP] frac={frac:.3f} th={th:.6f} flagged_frac={flagged_frac:.3f} "
            f"tp={tp} fp={fp} fn={fn} recall={rec:.6f} precision={prec:.6f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
