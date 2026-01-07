#!/usr/bin/env python3
"""Compute per-row tree-disagreement (std of per-tree margin contributions).

This is a practical proxy for *epistemic uncertainty* in a boosted tree ensemble:
if trees disagree a lot (large std), the model is likely out-of-distribution or
near decision boundaries.

We implement it efficiently using:
- `predict(pred_leaf=True)` to get the leaf node id per tree
- `get_dump(dump_format="json")` to map leaf node id -> leaf value per tree

Outputs:
- std_f32le: float32 little-endian, one value per row
Optionally also write model score/probabilities.

Example:
  python3 scripts/ml/score_xgb_tree_std_dense_f32le.py \
    --model-dir models/ieee_l1l2_full/ieee_l1 \
    --dense-file models/data/bench/ieee_20k.f32 --dim 432 \
    --out-std scores/l1_tree_std_20k.f32 --out-score scores/l1_scores_20k.f32
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass

import numpy as np

try:
    import xgboost as xgb
except Exception as e:
    raise SystemExit(f"xgboost import failed: {e}")


def die(msg: str) -> None:
    raise SystemExit(msg)


def read_f32le_dense(path: str, dim: int) -> np.ndarray:
    raw = np.fromfile(path, dtype="<f4")
    if raw.size % dim != 0:
        die(f"dense-file size mismatch: n_floats={raw.size} dim={dim}")
    return raw.reshape((-1, dim))


def load_feature_names(model_dir: str) -> list[str] | None:
    fp = os.path.join(model_dir, "feature_names.json")
    if not os.path.exists(fp):
        return None
    with open(fp, "r", encoding="utf-8") as f:
        names = json.load(f)
    if not isinstance(names, list):
        die(f"bad feature_names.json: expected list, got {type(names)}")
    return [str(x) for x in names]


def load_model_path(model_dir: str) -> str:
    for cand in ("ieee_xgb.ubj", "model.ubj", "xgb.ubj", "xgb.json"):
        p = os.path.join(model_dir, cand)
        if os.path.exists(p):
            return p
    # fallback
    p = os.path.join(model_dir, "xgb_model.json")
    if os.path.exists(p):
        return p
    die(f"no model found under: {model_dir}")


def _collect_leaf_values(node: dict, out: dict[int, float]) -> None:
    if "leaf" in node:
        out[int(node.get("nodeid", 0))] = float(node["leaf"])
        return
    for ch in node.get("children", []) or []:
        _collect_leaf_values(ch, out)


@dataclass
class LeafLookup:
    arr: np.ndarray  # float32, indexed by nodeid


def build_leaf_lookups(bst: "xgb.Booster") -> list[LeafLookup]:
    dumps = bst.get_dump(dump_format="json")
    lookups: list[LeafLookup] = []
    for t, s in enumerate(dumps):
        obj = json.loads(s)
        m: dict[int, float] = {}
        _collect_leaf_values(obj, m)
        if not m:
            die(f"tree {t} dump had no leaves")
        max_id = max(m.keys())
        arr = np.full((max_id + 1,), np.nan, dtype=np.float32)
        for k, v in m.items():
            arr[k] = np.float32(v)
        lookups.append(LeafLookup(arr=arr))
    return lookups


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--dense-file", required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--out-std", required=True)
    ap.add_argument("--out-score", default="")
    ap.add_argument("--batch-rows", type=int, default=50000)
    ap.add_argument("--validate-features", action="store_true")
    args = ap.parse_args()

    model_dir = str(args.model_dir)
    model_path = load_model_path(model_dir)
    feat_names = load_feature_names(model_dir)

    print(f"[TREE_STD] model={model_path}")
    print(f"[TREE_STD] dense={args.dense_file} dim={args.dim}")

    bst = xgb.Booster()
    bst.load_model(model_path)

    lookups = build_leaf_lookups(bst)
    n_trees = len(lookups)
    print(f"[TREE_STD] trees={n_trees}")

    X = read_f32le_dense(str(args.dense_file), int(args.dim))
    n = X.shape[0]

    out_std = np.empty((n,), dtype=np.float32)
    out_score = np.empty((n,), dtype=np.float32) if args.out_score else None

    bs = int(args.batch_rows)
    validate = bool(args.validate_features)

    for i in range(0, n, bs):
        j = min(n, i + bs)
        xb = X[i:j]
        dmat = xgb.DMatrix(xb, feature_names=feat_names)

        # leaf ids: (batch, trees)
        leaf = bst.predict(dmat, pred_leaf=True, validate_features=validate)
        # compute sum and sumsq of per-tree leaf values (margin contributions)
        s = np.zeros((j - i,), dtype=np.float64)
        ss = np.zeros((j - i,), dtype=np.float64)

        # Loop over trees; each step is vectorized indexing.
        for t in range(n_trees):
            ids = leaf[:, t].astype(np.int64, copy=False)
            arr = lookups[t].arr
            if ids.max(initial=0) >= arr.shape[0]:
                die(
                    f"leaf nodeid out of range in tree {t}: max_id={ids.max()} arr_len={arr.shape[0]}"
                )
            v = arr[ids].astype(np.float64, copy=False)
            if np.isnan(v).any():
                die(f"nan leaf value encountered in tree {t} (bad dump mapping)")
            s += v
            ss += v * v

        mean = s / float(n_trees)
        var = ss / float(n_trees) - mean * mean
        var = np.maximum(var, 0.0)
        out_std[i:j] = np.sqrt(var).astype(np.float32)

        if out_score is not None:
            out_score[i:j] = bst.predict(dmat, validate_features=validate).astype(
                np.float32
            )

    out_std.tofile(str(args.out_std))
    print(f"[TREE_STD] wrote std: {args.out_std} n={n}")
    print(
        f"[TREE_STD] std_stats: min={out_std.min():.6g} p50={np.quantile(out_std, 0.5):.6g} p95={np.quantile(out_std, 0.95):.6g} max={out_std.max():.6g}"
    )

    if out_score is not None:
        out_score.tofile(str(args.out_score))
        print(f"[TREE_STD] wrote score: {args.out_score} n={n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
