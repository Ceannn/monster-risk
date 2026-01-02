#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polars-first IEEE-CIS XGBoost trainer (L1/L2) with LOWER PEAK MEMORY for full training on WSL2.

Key memory tricks vs the earlier version:
1) Build X_train/X_valid once (float32), then build DMatrix/QuantileDMatrix ONCE and immediately `del X_*`.
2) Reuse the SAME dtrain/dvalid for L1 and L2 training (avoids building quantized caches twice).
3) Polars Lazy + streaming group_by to build categorical top-K maps without eager Series.value_counts().

Artifacts (compatible with your Rust runtime):
- ieee_xgb.ubj (preferred) + xgb_model.json (fallback) [+ ieee_xgb.bin optional]
- feature_names.json
- cat_maps.json.gz  (unknown/missing -> 0.0)
- policy.json       (L1 routing thresholds)
- train_meta.json

Usage:
  # dry run
  python3 scripts/train_ieee_xgb_polars_full.py --data-dir data/raw/ieee-cis --out-root models --sample 0.2

  # full
  MALLOC_ARENA_MAX=2 python3 scripts/train_ieee_xgb_polars_full.py --data-dir data/raw/ieee-cis --out-root models --sample 1.0

If full still tight, try:
  --drop-cols-prefix V
  --max-cat-unique 20000
  --topk 128
"""

from __future__ import annotations
import argparse
import gzip
import json
import sys
import gc
from pathlib import Path
from typing import Dict, Any, List, Tuple

def die(msg: str) -> None:
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(1)

def require(pkg: str):
    try:
        __import__(pkg)
    except Exception as e:
        die(f"Missing python package '{pkg}'. Install it first. Error: {e}")

require("polars")
require("xgboost")
require("sklearn")

import polars as pl
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--out-root", type=str, required=True)
    p.add_argument("--topk", type=int, default=256)
    p.add_argument("--max-cat-unique", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--valid-frac", type=float, default=0.2)
    p.add_argument("--sample", type=float, default=1.0)
    p.add_argument("--nthread", type=int, default=4,
                   help="Training threads. For WSL2, 4 is usually a good balance of speed & memory.")
    p.add_argument("--l2-route-frac", type=float, default=0.10)
    p.add_argument("--deny-frac", type=float, default=0.005)
    p.add_argument("--infer-schema", type=int, default=2000)
    p.add_argument("--drop-cols-prefix", type=str, default="",
                   help="Comma-separated prefixes to drop, e.g. 'V,M_'")
    return p.parse_args()

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))

def save_cat_maps_gz(path: Path, cat_maps: Dict[str, Dict[str, float]]) -> None:
    raw = json.dumps(cat_maps, ensure_ascii=False).encode("utf-8")
    with gzip.open(path, "wb") as f:
        f.write(raw)

def schema_names(lf: pl.LazyFrame) -> List[str]:
    return lf.collect_schema().names()

def load_lazy_tables(data_dir: Path, infer_schema: int) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
    tx_path = data_dir / "train_transaction.csv"
    id_path = data_dir / "train_identity.csv"
    if not tx_path.exists():
        die(f"Missing {tx_path}")
    if not id_path.exists():
        die(f"Missing {id_path}")

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

def apply_optional_drops(lf: pl.LazyFrame, drop_prefixes: List[str]) -> pl.LazyFrame:
    if not drop_prefixes:
        return lf
    cols = schema_names(lf)
    drop = []
    for c in cols:
        for pref in drop_prefixes:
            if pref and c.startswith(pref):
                drop.append(c)
                break
    if drop:
        print(f"[DROP] {len(drop)} columns by prefix: {drop_prefixes}")
        return lf.drop(drop)
    return lf

def sample_lazy(lf: pl.LazyFrame, frac: float, seed: int) -> pl.LazyFrame:
    if frac >= 0.999999:
        return lf
    mod = int(frac * 10000)
    mod = max(1, min(9999, mod))
    cols = schema_names(lf)
    if "TransactionID" in cols:
        return lf.filter((pl.col("TransactionID").hash(seed) % 10000) < mod)
    return lf.with_row_count("row_id").filter((pl.col("row_id").hash(seed) % 10000) < mod)

def compute_time_cutoff(lf: pl.LazyFrame, valid_frac: float) -> Tuple[float, bool]:
    cols = schema_names(lf)
    if "TransactionDT" not in cols:
        return 0.0, False
    dt = lf.select(pl.col("TransactionDT")).collect(engine="streaming")
    cutoff = float(dt.select(pl.col("TransactionDT").quantile(1.0 - valid_frac, "nearest")).item())
    return cutoff, True

def detect_utf8_columns(lf_train: pl.LazyFrame) -> List[str]:
    df = lf_train.head(1000).collect(engine="streaming")
    return [c for c, dt in zip(df.columns, df.dtypes) if dt == pl.Utf8]

def compute_cat_maps(lf_train: pl.LazyFrame, topk: int, max_cat_unique: int) -> Dict[str, Dict[str, float]]:
    cat_cols = detect_utf8_columns(lf_train)
    print(f"[CATS] Utf8 categorical cols detected: {len(cat_cols)}")

    cat_maps: Dict[str, Dict[str, float]] = {}

    for idx, col in enumerate(cat_cols):
        try:
            nuniq = int(lf_train.select(pl.col(col).n_unique()).collect(engine="streaming").item())
        except Exception:
            nuniq = max_cat_unique + 1

        if nuniq > max_cat_unique:
            print(f"  [SKIP] {col}: n_unique≈{nuniq} > {max_cat_unique}")
            continue

        vc = (
            lf_train
            .group_by(pl.col(col), maintain_order=False)
            .agg(pl.len().alias("__count"))
            .sort("__count", descending=True)
            .limit(topk)
            .collect(engine="streaming")
        )
        vals = vc[col].to_list()
        mapping = {str(v): float(i + 1) for i, v in enumerate(vals) if v is not None}
        cat_maps[col] = mapping

        if (idx + 1) % 10 == 0 or (idx + 1) == len(cat_cols):
            print(f"  built maps: {idx+1}/{len(cat_cols)}")

    return cat_maps

def encode_and_materialize(lf: pl.LazyFrame, cat_maps: Dict[str, Dict[str, float]], drop_cols: List[str]):
    # NumPy only here (final materialization); XGBoost depends on it anyway.
    import numpy as np

    lf2 = lf
    cols = schema_names(lf2)

    for col, mp in cat_maps.items():
        if col in cols:
            lf2 = lf2.with_columns(
                pl.col(col).replace_strict(mp, default=0.0).cast(pl.Float32)
            )

    for c in cols:
        if c in drop_cols:
            continue
        lf2 = lf2.with_columns(
            pl.col(c).cast(pl.Float32, strict=False).fill_null(float("nan"))
        )

    df = lf2.collect(engine="streaming")
    if "isFraud" not in df.columns:
        die("Missing isFraud after join.")

    y = df["isFraud"].cast(pl.Int8).to_numpy().reshape(-1)
    feat_cols = [c for c in df.columns if c not in drop_cols]
    X = df.select(feat_cols).to_numpy()
    X = X.astype(np.float32, copy=False)
    return X, y, feat_cols

def make_dmat(X, y, feature_names: List[str], max_bin: int = 256, ref=None):
    """
    Build a (Quantile)DMatrix.

    IMPORTANT (XGBoost requirement):
      If we use QuantileDMatrix for training, then evaluation QuantileDMatrix MUST be constructed
      with ref=dtrain so it reuses the same quantile cuts. Otherwise you get:
        "Training dataset should be used as a reference ..."
    """
    missing = float("nan")

    # Prefer QuantileDMatrix for hist to reduce memory in some setups.
    # Pass ref when building eval set.
    try:
        return xgb.QuantileDMatrix(
            X,
            label=y,
            feature_names=feature_names,
            missing=missing,
            max_bin=max_bin,
            ref=ref,
        )
    except TypeError:
        # Older XGBoost may not accept ref/max_bin on QuantileDMatrix.
        try:
            return xgb.QuantileDMatrix(
                X,
                label=y,
                feature_names=feature_names,
                missing=missing,
            )
        except Exception:
            return xgb.DMatrix(X, label=y, feature_names=feature_names, missing=missing)
    except Exception:
        return xgb.DMatrix(X, label=y, feature_names=feature_names, missing=missing)


def train_xgb_with_dmat(dtrain, dvalid, y_valid, params: Dict[str, Any],
                        num_boost_round: int, early_stopping_rounds: int):
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=50,
    )
    p = booster.predict(dvalid)
    auc = float(roc_auc_score(y_valid, p))
    ap = float(average_precision_score(y_valid, p))
    metrics = {"valid_auc": auc, "valid_ap": ap, "best_iteration": int(getattr(booster, "best_iteration", 0))}
    return booster, metrics, p

def compute_l1_policy_thresholds(p_valid, l2_route_frac: float, deny_frac: float) -> Dict[str, float]:
    import numpy as np
    l2_route_frac = float(np.clip(l2_route_frac, 0.01, 0.80))
    deny_frac = float(np.clip(deny_frac, 0.0001, 0.20))
    deny_th = float(np.quantile(p_valid, 1.0 - deny_frac))
    q_review = 1.0 - (l2_route_frac + deny_frac)
    q_review = float(np.clip(q_review, 0.05, 0.98))
    review_th = float(np.quantile(p_valid, q_review))
    review_th = min(review_th, deny_th - 1e-6)
    return {"review_threshold": review_th, "deny_threshold": deny_th}

def save_model_dir(model_dir: Path, booster: xgb.Booster, feature_names: List[str],
                   cat_maps: Dict[str, Dict[str, float]], policy: Dict[str, float], meta: Dict[str, Any]) -> None:
    ensure_dir(model_dir)
    save_json(model_dir / "feature_names.json", feature_names)
    save_cat_maps_gz(model_dir / "cat_maps.json.gz", cat_maps)
    save_json(model_dir / "policy.json", policy)
    save_json(model_dir / "train_meta.json", meta)

    ubj_path = model_dir / "ieee_xgb.ubj"
    json_path = model_dir / "xgb_model.json"
    bin_path = model_dir / "ieee_xgb.bin"

    saved = []
    try:
        booster.save_model(str(ubj_path))
        saved.append(str(ubj_path))
    except Exception as e:
        print(f"[WARN] save_model(.ubj) failed: {e}")

    booster.save_model(str(json_path))
    saved.append(str(json_path))

    try:
        booster.save_model(str(bin_path))
        saved.append(str(bin_path))
    except Exception:
        pass

    print(f"[SAVE] {model_dir} -> {', '.join(saved)}")

def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_root = Path(args.out_root)
    ensure_dir(out_root)

    if not (0.0 < args.sample <= 1.0):
        die("--sample must be in (0,1]")

    drop_prefixes = [x.strip() for x in args.drop_cols_prefix.split(",") if x.strip()]

    tx_lf, id_lf = load_lazy_tables(data_dir, args.infer_schema)
    tx_lf = apply_optional_drops(tx_lf, drop_prefixes)
    id_lf = apply_optional_drops(id_lf, drop_prefixes)

    tx_cols = schema_names(tx_lf)
    id_cols = schema_names(id_lf)

    if "TransactionID" not in tx_cols or "TransactionID" not in id_cols:
        die("Expected TransactionID in both transaction and identity tables.")
    if "isFraud" not in tx_cols:
        die("Expected isFraud in train_transaction.csv")

    tx_lf = sample_lazy(tx_lf, args.sample, args.seed)
    lf = tx_lf.join(id_lf, on="TransactionID", how="left")

    cutoff, use_time = compute_time_cutoff(lf, args.valid_frac)
    if use_time:
        lf_train = lf.filter(pl.col("TransactionDT") < cutoff)
        lf_valid = lf.filter(pl.col("TransactionDT") >= cutoff)
        print(f"[SPLIT] time-based: TransactionDT<{cutoff}")
    else:
        mod = int(args.valid_frac * 10000)
        mod = max(1, min(9999, mod))
        lf_valid = lf.filter((pl.col("TransactionID").hash(args.seed) % 10000) < mod)
        lf_train = lf.filter((pl.col("TransactionID").hash(args.seed) % 10000) >= mod)
        print(f"[SPLIT] hash-based valid_frac≈{args.valid_frac}")

    cat_maps = compute_cat_maps(lf_train, topk=args.topk, max_cat_unique=args.max_cat_unique)

    lf_cols = schema_names(lf)
    drop_cols = ["isFraud"]
    if "TransactionID" in lf_cols:
        drop_cols.append("TransactionID")

    print("[MATERIALIZE] train matrix (main allocation)")
    X_train, y_train, feat_names = encode_and_materialize(lf_train, cat_maps, drop_cols)
    print(f"  X_train={X_train.shape} features={len(feat_names)}")

    print("[MATERIALIZE] valid matrix")
    X_valid, y_valid, feat_names2 = encode_and_materialize(lf_valid, cat_maps, drop_cols)
    if feat_names2 != feat_names:
        die("Feature name mismatch between train and valid.")
    print(f"  X_valid={X_valid.shape}")

    # Build DMatrix ONCE, then drop X_* to reduce peak memory.
    print("[DMATRIX] building (Quantile)DMatrix and freeing raw matrices...")

    # Build training DMatrix first. If it's QuantileDMatrix, build valid with ref=dtrain.
    dtrain = make_dmat(X_train, y_train, feat_names, max_bin=256, ref=None)
    dvalid = make_dmat(X_valid, y_valid, feat_names, max_bin=256, ref=dtrain)

    # Free big raw matrices ASAP to lower peak RSS.
    del X_train, X_valid
    gc.collect()

    common = dict(
        objective="binary:logistic",
        eval_metric=["auc", "aucpr"],
        tree_method="hist",
        max_bin=256,
        nthread=int(args.nthread),
        seed=int(args.seed),
        predictor="cpu_predictor",
    )
    # Optional: class imbalance can help AP; uncomment if you want:
    # pos = max(1.0, float((y_train == 1).sum()))
    # neg = max(1.0, float((y_train == 0).sum()))
    # common["scale_pos_weight"] = neg / pos

    l1_params = dict(
        **common,
        max_depth=4,
        eta=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        reg_alpha=0.0,
    )
    l2_params = dict(
        **common,
        max_depth=7,
        eta=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        reg_lambda=1.0,
        reg_alpha=0.0,
    )

    print("[TRAIN] L1 (small) using shared DMatrix")
    l1_booster, l1_metrics, p_valid_l1 = train_xgb_with_dmat(
        dtrain, dvalid, y_valid, l1_params, num_boost_round=2000, early_stopping_rounds=100
    )
    print(f"[METRIC] L1 auc={l1_metrics['valid_auc']:.6f} ap={l1_metrics['valid_ap']:.6f} best_iter={l1_metrics['best_iteration']}")
    l1_policy = compute_l1_policy_thresholds(p_valid_l1, args.l2_route_frac, args.deny_frac)
    print(f"[POLICY] L1 review_th={l1_policy['review_threshold']:.6f} deny_th={l1_policy['deny_threshold']:.6f}")

    print("[TRAIN] L2 (large) using shared DMatrix")
    l2_booster, l2_metrics, _ = train_xgb_with_dmat(
        dtrain, dvalid, y_valid, l2_params, num_boost_round=6000, early_stopping_rounds=200
    )
    print(f"[METRIC] L2 auc={l2_metrics['valid_auc']:.6f} ap={l2_metrics['valid_ap']:.6f} best_iter={l2_metrics['best_iteration']}")

    l2_policy = {"review_threshold": 0.5, "deny_threshold": 0.9}

    l1_dir = out_root / "ieee_l1"
    l2_dir = out_root / "ieee_l2"

    save_model_dir(
        l1_dir, l1_booster, feat_names, cat_maps, l1_policy,
        meta={"kind":"L1","params":l1_params, **l1_metrics,
              "topk":args.topk, "sample":args.sample, "max_cat_unique": args.max_cat_unique},
    )
    save_model_dir(
        l2_dir, l2_booster, feat_names, cat_maps, l2_policy,
        meta={"kind":"L2","params":l2_params, **l2_metrics,
              "topk":args.topk, "sample":args.sample, "max_cat_unique": args.max_cat_unique},
    )

    print("\n[DONE]")
    print(f"  L1 -> {l1_dir}")
    print(f"  L2 -> {l2_dir}")
    print("Next: build isolated L1 pool + L2 pool and route by L1 policy thresholds.")

if __name__ == "__main__":
    main()
