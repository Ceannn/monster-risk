#!/usr/bin/env python3
"""Adversarial validation for IEEE-CIS features.

Goal: detect distribution shift / potential time leakage by training a classifier
that predicts which split a row came from (train vs test, or early vs late).

This is meant to be run BEFORE you start heavy "炼丹".

Outputs:
- AUC/AP for split classifier
- top drifting features by gain importance
- (optional) writes a CSV with importances

Example (train vs test):
  python3 scripts/ml/adversarial_validate_ieee_polars.py \
    --data-dir models/data/ieee \
    --mode train_vs_test \
    --sample-frac 0.2 \
    --out-csv results/adv_train_vs_test.csv

Example (time split within train):
  python3 scripts/ml/adversarial_validate_ieee_polars.py \
    --data-dir models/data/ieee \
    --mode early_vs_late \
    --valid-frac 0.2 \
    --sample-frac 0.3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import polars as pl
import xgboost as xgb


def die(msg: str) -> None:
    raise SystemExit(msg)


def schema_names(lf: pl.LazyFrame) -> List[str]:
    return lf.collect_schema().names()


def sample_lazy(lf: pl.LazyFrame, frac: float, seed: int) -> pl.LazyFrame:
    if frac >= 0.999999:
        return lf
    mod = int(frac * 10000)
    mod = max(1, min(9999, mod))
    cols = schema_names(lf)
    if "TransactionID" in cols:
        return lf.filter((pl.col("TransactionID").hash(seed) % 10000) < mod)
    return lf.with_row_count("row_id").filter(
        (pl.col("row_id").hash(seed) % 10000) < mod
    )


def apply_optional_drops(lf: pl.LazyFrame, drop_prefixes: List[str]) -> pl.LazyFrame:
    if not drop_prefixes:
        return lf
    cols = schema_names(lf)
    drop: List[str] = []
    for c in cols:
        for pref in drop_prefixes:
            if pref and c.startswith(pref):
                drop.append(c)
                break
    if drop:
        print(f"[DROP] {len(drop)} columns by prefix: {drop_prefixes}")
        return lf.drop(drop)
    return lf


def detect_utf8_columns(lf: pl.LazyFrame) -> List[str]:
    df = lf.head(2000).collect(engine="streaming")
    return [c for c, dt in zip(df.columns, df.dtypes) if dt == pl.Utf8]


def compute_cat_maps(
    lf: pl.LazyFrame, topk: int, max_cat_unique: int
) -> Dict[str, Dict[str, float]]:
    cat_cols = detect_utf8_columns(lf)
    print(f"[CATS] Utf8 categorical cols detected: {len(cat_cols)}")
    out: Dict[str, Dict[str, float]] = {}

    for idx, col in enumerate(cat_cols):
        try:
            nuniq = int(
                lf.select(pl.col(col).n_unique()).collect(engine="streaming").item()
            )
        except Exception:
            nuniq = max_cat_unique + 1

        if nuniq > max_cat_unique:
            # high-card columns are usually pure IDs; they will trivially leak time
            # but are not useful to keep in the final risk model either.
            print(f"  [SKIP] {col}: n_unique≈{nuniq} > {max_cat_unique}")
            continue

        vc = (
            lf.group_by(pl.col(col), maintain_order=False)
            .agg(pl.len().alias("__count"))
            .sort("__count", descending=True)
            .limit(topk)
            .collect(engine="streaming")
        )
        vals = vc[col].to_list()
        mapping = {str(v): float(i + 1) for i, v in enumerate(vals) if v is not None}
        out[col] = mapping

        if (idx + 1) % 10 == 0 or (idx + 1) == len(cat_cols):
            print(f"  built maps: {idx + 1}/{len(cat_cols)}")

    return out


def encode_and_materialize(
    lf: pl.LazyFrame, cat_maps: Dict[str, Dict[str, float]]
) -> Tuple[np.ndarray, List[str]]:
    cols = schema_names(lf)

    exprs: List[pl.Expr] = []
    feature_cols: List[str] = []

    for c in cols:
        if c in ("__adv_label", "isFraud"):
            continue
        if c in cat_maps:
            mapping = cat_maps[c]
            exprs.append(
                pl.col(c)
                .cast(pl.Utf8)
                .map_elements(
                    lambda v: float(mapping.get(str(v), 0.0)) if v is not None else 0.0,
                    return_dtype=pl.Float32,
                )
                .alias(c)
            )
        else:
            # numeric-ish
            exprs.append(
                pl.col(c).cast(pl.Float32, strict=False).fill_null(0.0).alias(c)
            )
        feature_cols.append(c)

    df = lf.select(exprs + [pl.col("__adv_label")]).collect(engine="streaming")
    X = df.select(feature_cols).to_numpy().astype(np.float32, copy=False)
    return X, feature_cols


def compute_auc_ap(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    # small helper: use xgb built-in metrics for consistency
    d = xgb.DMatrix(np.zeros((len(y), 1), dtype=np.float32), label=y)
    # hack: xgb doesn't expose metric functions directly; implement AUC with sklearn-like formula? avoid dependency.
    # We'll compute AUC via rank-sum (Mann–Whitney) and AP via precision-recall integration.
    y = y.astype(np.int32)
    p = p.astype(np.float64)

    # AUC (rank-sum)
    order = np.argsort(p)
    y_sorted = y[order]
    n_pos = int(y_sorted.sum())
    n_neg = int(len(y_sorted) - n_pos)
    if n_pos == 0 or n_neg == 0:
        auc = float("nan")
    else:
        ranks = np.arange(1, len(y_sorted) + 1)
        pos_ranks_sum = float(ranks[y_sorted == 1].sum())
        auc = (pos_ranks_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    # AP
    desc = np.argsort(-p)
    y2 = y[desc]
    tp = np.cumsum(y2)
    fp = np.cumsum(1 - y2)
    denom = tp + fp
    precision = np.divide(
        tp, denom, out=np.zeros_like(tp, dtype=np.float64), where=denom != 0
    )
    recall = tp / max(1, n_pos)
    # integrate precision over recall steps
    ap = 0.0
    prev_r = 0.0
    for pr, rc in zip(precision, recall):
        if rc > prev_r:
            ap += pr * (rc - prev_r)
            prev_r = rc
    return float(auc), float(ap)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument(
        "--mode",
        type=str,
        default="train_vs_test",
        choices=["train_vs_test", "early_vs_late"],
        help="train_vs_test uses train_* and test_* CSVs; early_vs_late splits train by TransactionDT quantile.",
    )
    p.add_argument("--infer-schema", type=int, default=200)
    p.add_argument("--sample-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--valid-frac", type=float, default=0.2)
    p.add_argument("--topk", type=int, default=200)
    p.add_argument("--max-cat-unique", type=int, default=500)
    p.add_argument(
        "--drop-prefix",
        action="append",
        default=[],
        help="Drop columns whose names start with this prefix.",
    )
    p.add_argument("--max-bin", type=int, default=256)
    p.add_argument("--nthread", type=int, default=8)
    p.add_argument("--out-csv", type=str, default="")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        die(f"missing data-dir: {data_dir}")

    # Load
    train_tx = data_dir / "train_transaction.csv"
    train_id = data_dir / "train_identity.csv"
    if not train_tx.exists() or not train_id.exists():
        die("missing train_transaction.csv / train_identity.csv")

    print(f"[SCAN] {train_tx}")
    tx_tr = pl.scan_csv(
        train_tx,
        infer_schema_length=int(args.infer_schema),
        ignore_errors=True,
        low_memory=True,
    )
    print(f"[SCAN] {train_id}")
    id_tr = pl.scan_csv(
        train_id,
        infer_schema_length=int(args.infer_schema),
        ignore_errors=True,
        low_memory=True,
    )

    lf_tr = tx_tr.join(id_tr, on="TransactionID", how="left")
    lf_tr = sample_lazy(lf_tr, float(args.sample_frac), int(args.seed))

    if args.mode == "train_vs_test":
        test_tx = data_dir / "test_transaction.csv"
        test_id = data_dir / "test_identity.csv"
        if not test_tx.exists() or not test_id.exists():
            die(
                "mode=train_vs_test but missing test_transaction.csv / test_identity.csv"
            )
        print(f"[SCAN] {test_tx}")
        tx_te = pl.scan_csv(
            test_tx,
            infer_schema_length=int(args.infer_schema),
            ignore_errors=True,
            low_memory=True,
        )
        print(f"[SCAN] {test_id}")
        id_te = pl.scan_csv(
            test_id,
            infer_schema_length=int(args.infer_schema),
            ignore_errors=True,
            low_memory=True,
        )
        lf_te = tx_te.join(id_te, on="TransactionID", how="left")
        lf_te = sample_lazy(lf_te, float(args.sample_frac), int(args.seed) + 1)

        lf_tr = lf_tr.with_columns(pl.lit(0).cast(pl.Int8).alias("__adv_label"))
        lf_te = lf_te.with_columns(pl.lit(1).cast(pl.Int8).alias("__adv_label"))
        lf = pl.concat([lf_tr, lf_te], how="diagonal")
    else:
        # early_vs_late within train by TransactionDT
        cols = schema_names(lf_tr)
        if "TransactionDT" not in cols:
            die("early_vs_late requires TransactionDT")
        dt = lf_tr.select(pl.col("TransactionDT")).collect(engine="streaming")
        cutoff = float(
            dt.select(
                pl.col("TransactionDT").quantile(
                    1.0 - float(args.valid_frac), "nearest"
                )
            ).item()
        )
        print(f"[SPLIT] early_vs_late cutoff TransactionDT={cutoff}")
        lf = lf_tr.with_columns(
            (pl.col("TransactionDT") >= cutoff).cast(pl.Int8).alias("__adv_label")
        )

    # Drops
    lf = apply_optional_drops(lf, list(args.drop_prefix))

    # Build cat maps on combined (so encoding doesn't introduce artificial shift)
    cat_maps = compute_cat_maps(lf, int(args.topk), int(args.max_cat_unique))

    # Encode
    X, feat_names = encode_and_materialize(lf, cat_maps)
    y = (
        lf.select(pl.col("__adv_label"))
        .collect(engine="streaming")
        .to_series()
        .to_numpy()
        .astype(np.int8)
    )

    # Shuffle/split
    n = len(y)
    rng = np.random.default_rng(int(args.seed))
    idx = np.arange(n)
    rng.shuffle(idx)
    n_valid = int(max(1, n * float(args.valid_frac)))
    vidx = idx[:n_valid]
    tidx = idx[n_valid:]

    X_train, y_train = X[tidx], y[tidx]
    X_valid, y_valid = X[vidx], y[vidx]

    print(
        f"[DATA] n={n} train={len(tidx)} valid={len(vidx)} pos_rate={float(y.mean()):.6f}"
    )

    dtrain = xgb.DMatrix(
        X_train, label=y_train.astype(np.float32), feature_names=feat_names
    )
    dvalid = xgb.DMatrix(
        X_valid, label=y_valid.astype(np.float32), feature_names=feat_names
    )

    params: Dict[str, Any] = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "eval_metric": ["logloss", "auc"],
        "max_depth": 4,
        "eta": 0.08,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "max_bin": int(args.max_bin),
        "nthread": int(args.nthread),
        "seed": int(args.seed),
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=400,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=40,
        verbose_eval=50,
    )

    p_valid = booster.predict(dvalid)
    auc, ap = compute_auc_ap(y_valid, p_valid)
    print(f"[ADV] auc={auc:.6f} ap={ap:.6f} best_iter={booster.best_iteration}")

    imp = booster.get_score(importance_type="gain")
    items = sorted(
        ((k, float(v)) for k, v in imp.items()), key=lambda kv: kv[1], reverse=True
    )

    print("[ADV] top drift features (gain):")
    for k, v in items[:30]:
        print(f"  {k}: {v:.6f}")

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            f.write("feature,gain\n")
            for k, v in items:
                f.write(f"{k},{v}\n")
        print(f"[ADV] wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
