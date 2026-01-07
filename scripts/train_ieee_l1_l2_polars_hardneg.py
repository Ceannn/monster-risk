#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IEEE-CIS: time-split L1/L2 XGBoost trainer (Polars-first, 6GB-friendly).

This script is meant to be the *research/training side* counterpart of your
Rust inference pipeline:

* Train/Valid split is strictly **time-based** (TransactionDT), never random.
* Feature mappings for categorical columns are computed ONLY on train (freeze).
* L2 training uses **Hard Negative Mining** (HNM): keep all fraud + only the
  "hard" non-fraud that L1 thinks are suspicious.

Outputs (compatible with your Rust runtime model_dir layout):
  - feature_names.json
  - cat_maps.json.gz
  - policy.json
  - train_meta.json
  - xgb_model.json (always)
  - ieee_xgb.ubj   (if supported by installed XGBoost)

Memory philosophy (WSL2 ~6GB):
  - Materialize X/y once in float32 (train + valid), then
  - HNM subset is written to **memmap** on disk to avoid a second peak.

Example:
  MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=1 \
    python3 scripts/train_ieee_l1_l2_polars_hardneg.py \
      --data-dir data/raw/ieee-cis \
      --out-root models \
      --nthread 4 \
      --sample 1.0
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def die(msg: str) -> None:
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(1)


def require(pkg: str) -> None:
    try:
        __import__(pkg)
    except Exception as e:
        die(f"Missing python package '{pkg}'. Install it first. Error: {e}")


require("polars")
require("xgboost")
require("sklearn")

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--out-root", type=str, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--valid-frac", type=float, default=0.2)
    p.add_argument("--sample", type=float, default=1.0)
    p.add_argument("--infer-schema", type=int, default=2000)

    # categorical encoding
    p.add_argument("--topk", type=int, default=256)
    p.add_argument("--max-cat-unique", type=int, default=50_000)
    p.add_argument(
        "--drop-cols-prefix",
        type=str,
        default="",
        help="Comma-separated prefixes to drop, e.g. 'V,M_'",
    )

    # training threads
    p.add_argument(
        "--nthread",
        type=int,
        default=4,
        help="Training threads. Under WSL2, 4 is a good speed/memory balance.",
    )

    # L1 "mad dog" params
    p.add_argument("--l1-rounds", type=int, default=200)
    p.add_argument("--l1-max-depth", type=int, default=3)
    p.add_argument("--l1-eta", type=float, default=0.1)
    p.add_argument("--l1-scale-pos-weight", type=float, default=50.0)
    p.add_argument("--l1-early-stop", type=int, default=50)

    # L1 policy calibration
    p.add_argument(
        "--l1-target-recall",
        type=float,
        default=0.99,
        help="Choose L1 review_threshold so that recall>=this on valid.",
    )
    p.add_argument(
        "--l1-deny-frac",
        type=float,
        default=0.005,
        help="Set L1 deny_threshold as top X%% of valid scores.",
    )

    p.add_argument(
        "--l1-route-frac",
        type=float,
        default=0.0,
        help=(
            "Optional: calibrate L1 review_threshold by route budget instead of recall. "
            "If >0, set review_th at quantile (1 - (route_frac + deny_frac))."
        ),
    )
    p.add_argument(
        "--l1-policy-mode",
        choices=["recall", "budget"],
        default="recall",
        help=(
            "How to pick L1 policy when both recall target and route budget are provided. "
            "recall: keep old behavior (highest th meeting target recall). "
            "budget: enforce --l1-route-frac budget (warn if recall is below target)."
        ),
    )

    # hard negative mining
    p.add_argument(
        "--hnm-neg-th",
        type=float,
        default=0.01,
        help="Keep negative samples with L1 score > this into L2 training.",
    )
    p.add_argument(
        "--hnm-neg-keep-frac",
        type=float,
        default=0.0,
        help=(
            "Optional: cap hard negatives by keeping only the top X%% negatives by L1 score. "
            "If >0, this takes precedence over --hnm-neg-th and makes L2 dataset size stable."
        ),
    )
    p.add_argument(
        "--memmap-dir",
        type=str,
        default=".cache/hnm",
        help="On-disk temp dir for memmap (keeps peak RSS low).",
    )

    # L2 "judge" params
    p.add_argument("--l2-rounds", type=int, default=3000)
    p.add_argument("--l2-max-depth", type=int, default=7)
    p.add_argument("--l2-eta", type=float, default=0.02)
    p.add_argument("--l2-scale-pos-weight", type=float, default=5.0)
    p.add_argument("--l2-early-stop", type=int, default=200)

    # optional: use fewer bins to reduce memory
    p.add_argument("--max-bin", type=int, default=256)

    # monotonic constraints (optional)
    p.add_argument(
        "--l1-mono",
        type=str,
        default="",
        help="Monotone constraints for L1, e.g. 'TransactionAmt:+1,C1:+1'.",
    )
    p.add_argument(
        "--l2-mono",
        type=str,
        default="",
        help="Monotone constraints for L2, same spec as --l1-mono.",
    )

    # distillation (optional): fine-tune L1 on L2 mined set using teacher soft labels
    p.add_argument(
        "--l1-distill-alpha",
        type=float,
        default=0.0,
        help="If >0, fine-tune L1 on mined set with soft labels: y_soft=(1-a)*y + a*L2_prob.",
    )
    p.add_argument(
        "--l1-distill-temperature",
        type=float,
        default=1.0,
        help="Teacher temperature (logit/T).",
    )
    p.add_argument(
        "--l1-distill-rounds",
        type=int,
        default=200,
        help="Extra boosting rounds for L1 distillation fine-tune.",
    )
    p.add_argument(
        "--l1-distill-early-stop",
        type=int,
        default=50,
        help="Early stopping rounds for L1 distillation fine-tune.",
    )

    # hard positives (optional): upweight positives that L1 thinks are low-risk
    p.add_argument(
        "--hnm-hard-pos-th",
        type=float,
        default=0.0,
        help="If >0, positives with L1 score < th are hard-pos.",
    )
    p.add_argument(
        "--hnm-hard-pos-mul",
        type=float,
        default=1.0,
        help="Multiply weight for hard positives in L2 training.",
    )

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


def load_lazy_tables(
    data_dir: Path, infer_schema: int
) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
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


def compute_time_cutoff(lf: pl.LazyFrame, valid_frac: float) -> float:
    cols = schema_names(lf)
    if "TransactionDT" not in cols:
        die(
            "TransactionDT missing: cannot do time-split (No Random Split is required)."
        )
    dt = lf.select(pl.col("TransactionDT")).collect(engine="streaming")
    cutoff = float(
        dt.select(pl.col("TransactionDT").quantile(1.0 - valid_frac, "nearest")).item()
    )
    return cutoff


def detect_utf8_columns(lf_train: pl.LazyFrame) -> List[str]:
    df = lf_train.head(1000).collect(engine="streaming")
    return [c for c, dt in zip(df.columns, df.dtypes) if dt == pl.Utf8]


def compute_cat_maps(
    lf_train: pl.LazyFrame, topk: int, max_cat_unique: int
) -> Dict[str, Dict[str, float]]:
    cat_cols = detect_utf8_columns(lf_train)
    print(f"[CATS] Utf8 categorical cols detected: {len(cat_cols)}")
    cat_maps: Dict[str, Dict[str, float]] = {}

    for idx, col in enumerate(cat_cols):
        try:
            nuniq = int(
                lf_train.select(pl.col(col).n_unique())
                .collect(engine="streaming")
                .item()
            )
        except Exception:
            nuniq = max_cat_unique + 1

        if nuniq > max_cat_unique:
            print(f"  [SKIP] {col}: n_unique≈{nuniq} > {max_cat_unique}")
            continue

        vc = (
            lf_train.group_by(pl.col(col), maintain_order=False)
            .agg(pl.len().alias("__count"))
            .sort("__count", descending=True)
            .limit(topk)
            .collect(engine="streaming")
        )
        vals = vc[col].to_list()
        mapping = {str(v): float(i + 1) for i, v in enumerate(vals) if v is not None}
        cat_maps[col] = mapping

        if (idx + 1) % 10 == 0 or (idx + 1) == len(cat_cols):
            print(f"  built maps: {idx + 1}/{len(cat_cols)}")

    return cat_maps


def encode_and_materialize(
    lf: pl.LazyFrame, cat_maps: Dict[str, Dict[str, float]], drop_cols: List[str]
):
    cols = schema_names(lf)

    nan = float("nan")
    exprs: List[pl.Expr] = []

    # 1) categorical freeze (train-only maps)
    for col, mp in cat_maps.items():
        if col in cols and col not in drop_cols:
            exprs.append(pl.col(col).replace_strict(mp, default=0.0).cast(pl.Float32))

    # 2) everything else -> float32 + NaN for missing
    cat_set = set(cat_maps.keys())
    for c in cols:
        if c in drop_cols:
            continue
        if c in cat_set:
            # already handled above
            continue
        exprs.append(pl.col(c).cast(pl.Float32, strict=False).fill_null(nan))

    df = lf.with_columns(exprs).collect(engine="streaming")
    if "isFraud" not in df.columns:
        die("Missing isFraud after join.")

    y = df["isFraud"].cast(pl.Int8).to_numpy().reshape(-1)
    feat_cols = [c for c in df.columns if c not in drop_cols]
    X = df.select(feat_cols).to_numpy().astype(np.float32, copy=False)
    return X, y, feat_cols


def make_dmat(
    X,
    y,
    feature_names: List[str],
    *,
    max_bin: int,
    ref=None,
):
    missing = float("nan")
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
        # Older versions
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


def train_xgb(
    dtrain,
    dvalid,
    y_valid,
    params: Dict[str, Any],
    rounds: int,
    early_stop: int,
    *,
    xgb_model=None,
):
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=rounds,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=early_stop,
        verbose_eval=50,
        xgb_model=xgb_model,
    )
    p = booster.predict(dvalid)
    auc = float(roc_auc_score(y_valid, p))
    ap = float(average_precision_score(y_valid, p))
    metrics = {
        "valid_auc": auc,
        "valid_ap": ap,
        "best_iteration": int(getattr(booster, "best_iteration", 0)),
    }
    return booster, metrics, p


def _clip_prob(p: np.ndarray) -> np.ndarray:
    return np.clip(p, 1e-8, 1.0 - 1e-8)


def _apply_temperature(prob: np.ndarray, temperature: float) -> np.ndarray:
    """Apply logit temperature scaling: sigmoid(logit(p)/T)."""
    t = float(temperature)
    if t <= 0.0 or abs(t - 1.0) < 1e-12:
        return prob
    p = _clip_prob(prob)
    logit = np.log(p / (1.0 - p))
    return 1.0 / (1.0 + np.exp(-logit / t))


def parse_mono_spec(spec: str, feature_names: List[str]) -> str | None:
    """Parse spec like 'featA:+1,featB:-1' into XGBoost monotone_constraints string."""
    s = (spec or "").strip()
    if not s:
        return None
    m: Dict[str, int] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            die(f"bad mono spec item (expect name:+1/-1): {part}")
        name, v = part.split(":", 1)
        name = name.strip()
        v = v.strip()
        if v not in ("+1", "1", "-1", "0"):
            die(f"bad mono direction for {name}: {v} (use +1/-1)")
        m[name] = int(v)
    vec: List[str] = []
    for fn in feature_names:
        vec.append(str(m.get(fn, 0)))
    # XGBoost expects a tuple-like string.
    return "(" + ",".join(vec) + ")"


def choose_l1_threshold_by_recall(
    p_valid: np.ndarray, y_valid: np.ndarray, target_recall: float
) -> float:
    """Pick the *highest* threshold t such that recall>=target_recall.

    Intuition:
      - Higher threshold => more samples PASS (fewer go to L2), but recall drops.
      - So we want the earliest point that still meets recall.
    """
    y = (y_valid == 1).astype(np.int8)
    total_pos = int(y.sum())
    if total_pos == 0:
        die("Valid set has no positive samples; cannot calibrate recall.")

    order = np.argsort(p_valid)[::-1]
    p_s = p_valid[order]
    y_s = y[order]

    tp_cum = np.cumsum(y_s)
    recall = tp_cum / float(total_pos)

    # We need recall >= target; as we move forward (lower threshold), recall increases.
    # Find the first index where recall >= target.
    idx = int(np.searchsorted(recall, target_recall, side="left"))
    if idx >= len(p_s):
        idx = len(p_s) - 1
    t = float(p_s[idx])
    return t


def threshold_for_top_frac_desc(scores: np.ndarray, frac: float) -> float:
    """Return threshold t so that roughly top frac samples satisfy score>=t.

    Uses an order-statistic (np.partition) to avoid full sort.
    """
    n = int(scores.shape[0])
    if n <= 0:
        die("Empty score array; cannot calibrate threshold.")
    f = float(frac)
    if f <= 0.0:
        return float("inf")
    if f >= 1.0:
        return float(np.min(scores))
    k = int(np.ceil(f * n))
    k = max(1, min(n, k))
    # kth largest == element at index n-k in ascending partition
    t = float(np.partition(scores, n - k)[n - k])
    return t


def pr_recall_at_threshold(scores: np.ndarray, y: np.ndarray, th: float):
    yb = y == 1
    total_pos = int(yb.sum())
    flagged = scores >= th
    tp = int((flagged & yb).sum())
    fp = int((flagged & (~yb)).sum())
    fn = total_pos - tp
    prec = float(tp / max(1, (tp + fp)))
    rec = float(tp / max(1, (tp + fn)))
    frac = float(flagged.mean())
    return {
        "flagged_frac": frac,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec,
    }


def print_l1_frontier(
    scores: np.ndarray, y: np.ndarray, deny_frac: float
) -> List[Dict[str, float]]:
    """Print a small recall@budget table for quick sanity."""
    pts: List[Dict[str, float]] = []
    for rf in [0.02, 0.05, 0.10, 0.20, 0.30, 0.50]:
        flagged = min(0.999999, max(0.0, float(rf) + float(deny_frac)))
        th = threshold_for_top_frac_desc(scores, flagged)
        m = pr_recall_at_threshold(scores, y, th)
        pts.append(
            {
                "route_frac": float(rf),
                "deny_frac": float(deny_frac),
                "review_th": float(th),
                "flagged_frac": float(m["flagged_frac"]),
                "flag_recall": float(m["recall"]),
                "flag_precision": float(m["precision"]),
            }
        )
    print("[FRONTIER] L1 recall/precision vs route budget (valid)")
    for it in pts:
        print(
            f"  rf={it['route_frac']:.3f} review_th={it['review_th']:.6f} "
            f"flagged={it['flagged_frac']:.3f} recall={it['flag_recall']:.6f} prec={it['flag_precision']:.6f}"
        )
    return pts


def subset_to_memmap(
    X: np.ndarray, y: np.ndarray, mask: np.ndarray, out_dir: Path
) -> Tuple[np.memmap, np.ndarray]:
    ensure_dir(out_dir)
    n_sel = int(mask.sum())
    if n_sel <= 0:
        die(
            "Hard-negative mining produced 0 samples; check --hnm-neg-th or --hnm-neg-keep-frac."
        )
    n_feat = int(X.shape[1])

    x_path = out_dir / "X_l2.f32"
    y_path = out_dir / "y_l2.i8"

    X_mm = np.memmap(str(x_path), dtype=np.float32, mode="w+", shape=(n_sel, n_feat))
    y_out = np.memmap(str(y_path), dtype=np.int8, mode="w+", shape=(n_sel,))

    w = 0
    block = 8192
    for start in range(0, X.shape[0], block):
        end = min(X.shape[0], start + block)
        m = mask[start:end]
        if not m.any():
            continue
        xb = X[start:end][m]
        yb = y[start:end][m]
        X_mm[w : w + xb.shape[0], :] = xb
        y_out[w : w + xb.shape[0]] = yb.astype(np.int8, copy=False)
        w += xb.shape[0]

    if w != n_sel:
        die(f"memmap write mismatch: wrote={w} expect={n_sel}")

    X_mm.flush()
    y_out.flush()
    return X_mm, np.asarray(y_out)


def save_model_dir(
    model_dir: Path,
    booster: xgb.Booster,
    feature_names: List[str],
    cat_maps: Dict[str, Dict[str, float]],
    policy: Dict[str, float],
    meta: Dict[str, Any],
) -> None:
    ensure_dir(model_dir)
    save_json(model_dir / "feature_names.json", feature_names)
    save_cat_maps_gz(model_dir / "cat_maps.json.gz", cat_maps)
    save_json(model_dir / "policy.json", policy)
    save_json(model_dir / "train_meta.json", meta)

    ubj_path = model_dir / "ieee_xgb.ubj"
    json_path = model_dir / "xgb_model.json"

    saved: List[str] = []
    try:
        booster.save_model(str(ubj_path))
        saved.append(str(ubj_path))
    except Exception as e:
        print(f"[WARN] save_model(.ubj) failed: {e}")

    booster.save_model(str(json_path))
    saved.append(str(json_path))
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

    cutoff = compute_time_cutoff(lf, args.valid_frac)
    lf_train = lf.filter(pl.col("TransactionDT") < cutoff)
    lf_valid = lf.filter(pl.col("TransactionDT") >= cutoff)
    print(f"[SPLIT] time-based: TransactionDT<{cutoff}")

    # ---- feature freeze: build categorical maps on TRAIN ONLY ----
    cat_maps = compute_cat_maps(
        lf_train, topk=args.topk, max_cat_unique=args.max_cat_unique
    )

    lf_cols = schema_names(lf)
    drop_cols = ["isFraud"]
    if "TransactionID" in lf_cols:
        drop_cols.append("TransactionID")

    print("[MATERIALIZE] train matrix")
    X_train, y_train, feat_names = encode_and_materialize(lf_train, cat_maps, drop_cols)
    print(f"  X_train={X_train.shape} features={len(feat_names)}")

    print("[MATERIALIZE] valid matrix")
    X_valid, y_valid, feat_names2 = encode_and_materialize(
        lf_valid, cat_maps, drop_cols
    )
    if feat_names2 != feat_names:
        die("Feature name mismatch between train and valid.")
    print(f"  X_valid={X_valid.shape}")

    # ---- build DMatrix (once) ----
    print("[DMATRIX] building (Quantile)DMatrix")
    dtrain = make_dmat(X_train, y_train, feat_names, max_bin=args.max_bin, ref=None)
    dvalid = make_dmat(X_valid, y_valid, feat_names, max_bin=args.max_bin, ref=dtrain)

    common = dict(
        objective="binary:logistic",
        eval_metric=["auc", "aucpr"],
        tree_method="hist",
        max_bin=int(args.max_bin),
        nthread=int(args.nthread),
        seed=int(args.seed),
        predictor="cpu_predictor",
    )

    # ---- L1 ----
    l1_params = dict(
        **common,
        max_depth=int(args.l1_max_depth),
        eta=float(args.l1_eta),
        scale_pos_weight=float(args.l1_scale_pos_weight),
        subsample=0.8,
        colsample_bytree=0.8,
    )
    mono = parse_mono_spec(str(args.l1_mono), feat_names)
    if mono is not None:
        l1_params["monotone_constraints"] = mono
        print(f"[TRAIN] L1 monotone_constraints enabled: {args.l1_mono}")
    print("[TRAIN] L1 (mad dog)")
    l1_booster, l1_metrics, p_valid_l1 = train_xgb(
        dtrain,
        dvalid,
        y_valid,
        l1_params,
        rounds=int(args.l1_rounds),
        early_stop=int(args.l1_early_stop),
    )
    print(
        f"[METRIC] L1 auc={l1_metrics['valid_auc']:.6f} ap={l1_metrics['valid_ap']:.6f} best_iter={l1_metrics['best_iteration']}"
    )

    # quick budget/recall table for sanity
    l1_frontier = print_l1_frontier(
        p_valid_l1, y_valid, deny_frac=float(args.l1_deny_frac)
    )
    base_l1_metrics = dict(l1_metrics)
    base_l1_frontier = list(l1_frontier)

    # Deny is always top deny_frac of the score distribution.
    deny_th = threshold_for_top_frac_desc(p_valid_l1, float(args.l1_deny_frac))
    base_deny_th = float(deny_th)

    # Review can be recall-calibrated (old) and/or budget-calibrated (new).
    recall_review_th = choose_l1_threshold_by_recall(
        p_valid_l1, y_valid, target_recall=float(args.l1_target_recall)
    )
    base_recall_review_th = float(recall_review_th)
    budget_review_th = float("nan")
    if float(args.l1_route_frac) > 0.0:
        flagged = min(0.999999, float(args.l1_route_frac) + float(args.l1_deny_frac))
        budget_review_th = threshold_for_top_frac_desc(p_valid_l1, flagged)
    base_budget_review_th = (
        float(budget_review_th) if float(args.l1_route_frac) > 0.0 else None
    )

    if args.l1_policy_mode == "budget" and float(args.l1_route_frac) > 0.0:
        review_th = float(budget_review_th)
        m = pr_recall_at_threshold(p_valid_l1, y_valid, review_th)
        if float(m["recall"]) + 1e-12 < float(args.l1_target_recall):
            print(
                f"[WARN] L1 recall below target under budget: got={m['recall']:.6f} want>={float(args.l1_target_recall):.6f} "
                f"(route_frac={float(args.l1_route_frac):.3f} deny_frac={float(args.l1_deny_frac):.3f})"
            )
    else:
        review_th = float(recall_review_th)

    if deny_th <= review_th:
        deny_th = min(0.999999, review_th + 1e-6)

    l1_policy = {"review_threshold": float(review_th), "deny_threshold": float(deny_th)}
    print(
        f"[POLICY] L1 selected: mode={args.l1_policy_mode} review_th={review_th:.6f} deny_th={deny_th:.6f}"
    )
    if float(args.l1_route_frac) > 0.0:
        print(
            f"[POLICY] L1 candidates: recall_review_th={float(recall_review_th):.6f} budget_review_th={float(budget_review_th):.6f}"
        )

    # ---- Hard Negative Mining (Train set) ----
    print("[HNM] predict L1 on TRAIN")
    p_train_l1 = l1_booster.predict(dtrain)

    mask_pos = y_train == 1
    if float(args.hnm_neg_keep_frac) > 0.0:
        neg_scores = p_train_l1[y_train == 0]
        th_hnm = threshold_for_top_frac_desc(neg_scores, float(args.hnm_neg_keep_frac))
        mask_hard_neg = (y_train == 0) & (p_train_l1 >= th_hnm)
        print(
            f"[HNM] mode=top_frac neg_keep_frac={float(args.hnm_neg_keep_frac):.3f} -> neg_th={float(th_hnm):.6f}"
        )
    else:
        mask_hard_neg = (y_train == 0) & (p_train_l1 > float(args.hnm_neg_th))
    mask_l2 = mask_pos | mask_hard_neg
    n_pos = int(mask_pos.sum())
    n_hn = int(mask_hard_neg.sum())
    print(f"[HNM] keep: pos={n_pos} hard_neg={n_hn} total={int(mask_l2.sum())}")


# Optional: upweight "hard positives" (fraud that L1 thinks are low-risk).
w_l2 = None
if float(args.hnm_hard_pos_th) > 0.0 and float(args.hnm_hard_pos_mul) > 1.0:
    hard_pos = (y_train == 1) & (p_train_l1 < float(args.hnm_hard_pos_th)) & mask_l2
    n_hp = int(hard_pos.sum())
    w_full = np.ones_like(y_train, dtype=np.float32)
    w_full[hard_pos] *= float(args.hnm_hard_pos_mul)
    w_l2 = w_full[mask_l2]
    print(
        f"[HNM] hard_pos_weight: th={float(args.hnm_hard_pos_th):.6f} mul={float(args.hnm_hard_pos_mul):.3f} n_hard_pos={n_hp}"
    )

    # Free big DMatrix caches before building L2 data (reduces peak)
    del dtrain
    gc.collect()

    memmap_dir = Path(args.memmap_dir)
    X_l2_mm, y_l2 = subset_to_memmap(X_train, y_train, mask_l2, memmap_dir)

    # Free full train matrix ASAP (largest resident block)
    del X_train
    gc.collect()

    # Build L2 DMatrix on the smaller mined set
    print("[DMATRIX] building L2 train DMatrix from memmap")
    dtrain_l2 = make_dmat(X_l2_mm, y_l2, feat_names, max_bin=args.max_bin, ref=None)
    if w_l2 is not None:
        try:
            dtrain_l2.set_weight(w_l2)
        except Exception as e:
            print(f"[WARN] failed to set L2 weights: {e}")
    dvalid_l2 = make_dmat(
        X_valid, y_valid, feat_names, max_bin=args.max_bin, ref=dtrain_l2
    )

    # Free valid raw matrix too once we have dvalid_l2
    del X_valid
    gc.collect()

    # ---- L2 ----
    l2_params = dict(
        **common,
        max_depth=int(args.l2_max_depth),
        eta=float(args.l2_eta),
        scale_pos_weight=float(args.l2_scale_pos_weight),
        subsample=0.7,
        colsample_bytree=0.5,
        gamma=0.1,
        min_child_weight=5,
    )
    mono2 = parse_mono_spec(str(args.l2_mono), feat_names)
    if mono2 is not None:
        l2_params["monotone_constraints"] = mono2
        print(f"[TRAIN] L2 monotone_constraints enabled: {args.l2_mono}")
    print("[TRAIN] L2 (judge) on mined set")
    l2_booster, l2_metrics, p_valid_l2 = train_xgb(
        dtrain_l2,
        dvalid_l2,
        y_valid,
        l2_params,
        rounds=int(args.l2_rounds),
        early_stop=int(args.l2_early_stop),
    )
    print(
        f"[METRIC] L2 auc={l2_metrics['valid_auc']:.6f} ap={l2_metrics['valid_ap']:.6f} best_iter={l2_metrics['best_iteration']}"
    )

    # Simple default L2 policy; you can later calibrate by precision@recall.
    l2_policy = {"review_threshold": 0.5, "deny_threshold": 0.9}


# ---- Optional: L1 <- L2 distillation (Privileged Information / Teacher routing) ----
distill_meta = None
if float(args.l1_distill_alpha) > 0.0:
    alpha = float(args.l1_distill_alpha)
    if not (0.0 < alpha < 1.0):
        die("--l1-distill-alpha must be in (0,1).")
    print(
        f"[DISTILL] enable: alpha={alpha:.3f} temperature={float(args.l1_distill_temperature):.3f} "
        f"rounds={int(args.l1_distill_rounds)} early_stop={int(args.l1_distill_early_stop)}"
    )

    # Teacher: L2 probabilities on the mined set (train)
    teacher_p = l2_booster.predict(dtrain_l2)
    teacher_p = _apply_temperature(teacher_p, float(args.l1_distill_temperature))

    # Soft label for L1 fine-tune: y_soft = (1-a)*y + a*teacher
    y_soft = (1.0 - alpha) * y_l2.astype(np.float32) + alpha * teacher_p.astype(
        np.float32
    )

    # Reuse the mined DMatrix but swap labels (saves memory)
    try:
        dtrain_l2.set_label(y_soft)
    except Exception as e:
        die(f"failed to set soft labels for distillation: {e}")

    # Continue boosting from the base L1 model, but train on mined set (hard cases)
    l1_booster, l1d_metrics, p_valid_l1 = train_xgb(
        dtrain_l2,
        dvalid_l2,
        y_valid,
        l1_params,
        rounds=int(args.l1_distill_rounds),
        early_stop=int(args.l1_distill_early_stop),
        xgb_model=l1_booster,
    )
    print(
        f"[METRIC] L1(distill) auc={l1d_metrics['valid_auc']:.6f} ap={l1d_metrics['valid_ap']:.6f} best_iter={l1d_metrics['best_iteration']}"
    )

    # Recompute frontier + policy on the new L1 outputs (valid set)
    l1_frontier = print_l1_frontier(
        p_valid_l1, y_valid, deny_frac=float(args.l1_deny_frac)
    )
    deny_th = threshold_for_top_frac_desc(p_valid_l1, float(args.l1_deny_frac))

    recall_review_th = choose_l1_threshold_by_recall(
        p_valid_l1, y_valid, target_recall=float(args.l1_target_recall)
    )
    budget_review_th = float("nan")
    if float(args.l1_route_frac) > 0.0:
        flagged = min(0.999999, float(args.l1_route_frac) + float(args.l1_deny_frac))
        budget_review_th = threshold_for_top_frac_desc(p_valid_l1, flagged)

    if args.l1_policy_mode == "budget" and float(args.l1_route_frac) > 0.0:
        review_th = float(budget_review_th)
        m = pr_recall_at_threshold(p_valid_l1, y_valid, review_th)
        if float(m["recall"]) + 1e-12 < float(args.l1_target_recall):
            print(
                f"[WARN] L1 recall below target under budget (after distill): got={m['recall']:.6f} want>={float(args.l1_target_recall):.6f} "
                f"(route_frac={float(args.l1_route_frac):.3f} deny_frac={float(args.l1_deny_frac):.3f})"
            )
    else:
        review_th = float(recall_review_th)

    if deny_th <= review_th:
        deny_th = min(0.999999, review_th + 1e-6)

    l1_policy = {"review_threshold": float(review_th), "deny_threshold": float(deny_th)}
    print(
        f"[POLICY] L1 selected (after distill): mode={args.l1_policy_mode} review_th={review_th:.6f} deny_th={deny_th:.6f}"
    )
    if float(args.l1_route_frac) > 0.0:
        print(
            f"[POLICY] L1 candidates (after distill): recall_review_th={float(recall_review_th):.6f} budget_review_th={float(budget_review_th):.6f}"
        )

    distill_meta = {
        "alpha": alpha,
        "temperature": float(args.l1_distill_temperature),
        "rounds": int(args.l1_distill_rounds),
        "early_stop": int(args.l1_distill_early_stop),
        "teacher": "L2_mined_set",
        "metrics": l1d_metrics,
    }

    # ---- save ----
    l1_dir = out_root / "ieee_l1"
    l2_dir = out_root / "ieee_l2"

    # persist frontier table for downstream policy calibration/debugging
    ensure_dir(l1_dir)
    frontier_path = l1_dir / "valid_frontier.csv"
    with open(frontier_path, "w", encoding="utf-8") as f:
        f.write(
            "route_frac,deny_frac,review_th,flagged_frac,flag_recall,flag_precision\n"
        )
        for it in l1_frontier:
            f.write(
                f"{it['route_frac']},{it['deny_frac']},{it['review_th']},{it['flagged_frac']},{it['flag_recall']},{it['flag_precision']}\n"
            )
    print(f"[SAVE] {frontier_path}")

    save_model_dir(
        l1_dir,
        l1_booster,
        feat_names,
        cat_maps,
        l1_policy,
        meta={
            "kind": "L1",
            "params": l1_params,
            **l1_metrics,
            "target_recall": float(args.l1_target_recall),
            "deny_frac": float(args.l1_deny_frac),
            "route_frac": float(args.l1_route_frac),
            "policy_mode": str(args.l1_policy_mode),
            "recall_review_th": float(recall_review_th),
            "budget_review_th": float(budget_review_th)
            if float(args.l1_route_frac) > 0.0
            else None,
            "base": {
                "metrics": base_l1_metrics,
                "deny_th": base_deny_th,
                "recall_review_th": base_recall_review_th,
                "budget_review_th": base_budget_review_th,
            },
            "distill": distill_meta,
            "monotone_constraints_spec": str(args.l1_mono)
            if str(args.l1_mono).strip()
            else None,
            "frontier_csv": str(frontier_path),
            "topk": int(args.topk),
            "sample": float(args.sample),
            "max_cat_unique": int(args.max_cat_unique),
        },
    )
    save_model_dir(
        l2_dir,
        l2_booster,
        feat_names,
        cat_maps,
        l2_policy,
        meta={
            "kind": "L2",
            "params": l2_params,
            **l2_metrics,
            "hnm_neg_th": float(args.hnm_neg_th),
            "hnm_neg_keep_frac": float(args.hnm_neg_keep_frac),
            "hnm_pos": n_pos,
            "hnm_hard_neg": n_hn,
            "hnm_hard_pos_th": float(args.hnm_hard_pos_th),
            "hnm_hard_pos_mul": float(args.hnm_hard_pos_mul),
            "topk": int(args.topk),
            "monotone_constraints_spec": str(args.l2_mono)
            if str(args.l2_mono).strip()
            else None,
            "sample": float(args.sample),
            "max_cat_unique": int(args.max_cat_unique),
        },
    )

    print("\n[DONE]")
    print(f"  L1 -> {l1_dir}")
    print(f"  L2 -> {l2_dir}")
    print(f"  HNM cache -> {memmap_dir} (safe to delete)")


if __name__ == "__main__":
    main()
