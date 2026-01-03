#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate IEEE-CIS request corpus as JSONL (NDJSON) for risk-bench.

- Reads feature_names.json from your model dir, so the output matches your Rust runtime expectations.
- Joins transaction + identity on TransactionID.
- Writes NDJSON in streaming mode via LazyFrame sink (low peak memory).
- Optional stratified sampling from train to increase L2 triggers (fraud_frac).

Examples:
  # realistic-ish corpus from test (no label)
  python3 scripts/make_ieee_jsonl_polars.py \
    --data-dir data/raw/ieee-cis --model-dir models/ieee_l1 \
    --split test --n 20000 --out data/bench/ieee_20k.jsonl

  # debug corpus that triggers L2 more often (oversample fraud from train)
  python3 scripts/make_ieee_jsonl_polars.py \
    --data-dir data/raw/ieee-cis --model-dir models/ieee_l1 \
    --split train --n 20000 --fraud-frac 0.2 --out data/bench/ieee_20k_posmix.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import polars as pl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, type=str)
    p.add_argument(
        "--model-dir",
        required=True,
        type=str,
        help="models/ieee_l1 or models/ieee_l2 (needs feature_names.json)",
    )
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--infer-schema", type=int, default=2000)
    p.add_argument(
        "--fraud-frac",
        type=float,
        default=0.0,
        help="Only for split=train. E.g. 0.2 means 20%% fraud rows in corpus (for L2 trigger debugging).",
    )
    p.add_argument("--out", required=True, type=str)
    return p.parse_args()


def load_feature_names(model_dir: Path) -> List[str]:
    fp = model_dir / "feature_names.json"
    if not fp.exists():
        raise SystemExit(f"[FATAL] missing {fp}")
    return json.loads(fp.read_text())


def scan_tables(
    data_dir: Path, split: str, infer_schema: int
) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
    if split == "train":
        tx_path = data_dir / "train_transaction.csv"
        id_path = data_dir / "train_identity.csv"
    else:
        tx_path = data_dir / "test_transaction.csv"
        id_path = data_dir / "test_identity.csv"

    if not tx_path.exists():
        raise SystemExit(f"[FATAL] missing {tx_path}")
    if not id_path.exists():
        raise SystemExit(f"[FATAL] missing {id_path}")

    print(f"[SCAN] {tx_path}")
    tx = pl.scan_csv(
        tx_path,
        infer_schema_length=infer_schema,
        ignore_errors=True,
        low_memory=True,
    )
    print(f"[SCAN] {id_path}")
    ident = pl.scan_csv(
        id_path,
        infer_schema_length=infer_schema,
        ignore_errors=True,
        low_memory=True,
    )
    return tx, ident


def schema_names(lf: pl.LazyFrame) -> List[str]:
    # cheap in polars>=1.x: collect_schema
    return lf.collect_schema().names()


def approx_take_n_by_hash(lf: pl.LazyFrame, n: int, seed: int) -> pl.LazyFrame:
    """Low-memory approximate sampling: hash filter + head(n)."""
    if n <= 0:
        return lf.head(0)

    cols = schema_names(lf)
    if "TransactionID" not in cols:
        # fallback: just head
        return lf.head(n)

    # count is scalar; OK
    total = int(lf.select(pl.len()).collect(engine="streaming").item())
    if n >= total:
        return lf

    # keep a bit extra then head(n) to cap precisely
    frac = min(1.0, (n / max(1, total)) * 1.6)
    mod = 1_000_000
    thr = int(frac * mod)
    thr = max(1, min(mod - 1, thr))

    return lf.filter((pl.col("TransactionID").hash(seed) % mod) < thr).head(n)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    feat_names = load_feature_names(model_dir)
    print(f"[MODEL] features={len(feat_names)} from {model_dir / 'feature_names.json'}")

    tx, ident = scan_tables(data_dir, args.split, args.infer_schema)
    # join
    lf = tx.join(ident, on="TransactionID", how="left")

    cols_all = set(schema_names(lf))

    # remove label/id keys from features if present
    drop = {"TransactionID", "isFraud"}
    use_cols = [c for c in feat_names if c in cols_all and c not in drop]

    missing = [c for c in feat_names if c not in cols_all and c not in drop]
    if missing:
        print(
            f"[WARN] {len(missing)} feature columns not found in this split; they will be omitted (runtime should treat missing as NaN)."
        )
        print(f"       example missing: {missing[:8]}")

    if args.split == "train" and args.fraud_frac > 0.0:
        ff = float(args.fraud_frac)
        ff = max(0.0, min(1.0, ff))
        n_pos = int(args.n * ff)
        n_neg = args.n - n_pos
        print(
            f"[SAMPLE] stratified train: n={args.n} fraud_frac={ff:.3f} -> pos={n_pos} neg={n_neg}"
        )

        lf_pos = lf.filter(pl.col("isFraud") == 1)
        lf_neg = lf.filter(pl.col("isFraud") == 0)

        lf_pos_s = approx_take_n_by_hash(lf_pos, n_pos, args.seed + 11)
        lf_neg_s = approx_take_n_by_hash(lf_neg, n_neg, args.seed + 23)

        lf_out = pl.concat([lf_pos_s, lf_neg_s], how="vertical_relaxed")
    else:
        print(f"[SAMPLE] hash+head: n={args.n} seed={args.seed}")
        lf_out = approx_take_n_by_hash(lf, args.n, args.seed)

    # wrap into {"features": {...}}
    features_struct = pl.struct([pl.col(c) for c in use_cols]).alias("features")
    lf_jsonl = lf_out.select(features_struct)

    print(f"[WRITE] {out_path} (ndjson) ...")
    # Polars streaming sink (low memory). If your polars is old, see fallback notes below.
    lf_jsonl.sink_ndjson(out_path)  # streaming by design in modern polars

    print("[DONE]")
    print(f"  file={out_path}")
    print("  sanity:")
    print(
        f"    head:  python3 - <<'PY'\nimport json\np=r'{out_path}'\nprint(json.loads(open(p).readline())['features'].keys().__len__())\nPY"
    )


if __name__ == "__main__":
    main()
