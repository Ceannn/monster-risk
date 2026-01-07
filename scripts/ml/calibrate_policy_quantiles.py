#!/usr/bin/env python3
"""Calibrate policy.json thresholds by offline score quantiles.

This script turns “we want L2 to trigger ~X%” into a reproducible artifact:
- Reads a score distribution (one score per row)
- Picks quantile thresholds
- Writes model_dir/policy.json + model_dir/policy_calibration.json

Key idea (for cascade routing):
- deny_th controls how many samples are hard-denied at this layer
- review_th controls how many samples are routed to the next layer

If we choose:
  deny_th   = quantile(1 - deny_frac)
  review_th = quantile(1 - (route_frac + deny_frac))
then (approximately):
  deny_frac  ~= P(score >= deny_th)
  route_frac ~= P(review_th <= score < deny_th)

Optional (label-aware) mode:
- Provide --labels-u8 and the script will print recall/precision for:
  * flagged = score >= review_th (routed OR denied at this layer)
  * deny    = score >= deny_th
- Provide --min-recall to enforce a minimum recall for flagged.

Important:
- Enforcing recall may require lowering review_th, which increases route_frac.
  That's the compute-vs-recall trade-off in cascade routing.

Deps:
  pip install numpy
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def die(msg: str, code: int = 2) -> "NoReturn":
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


@dataclass
class Policy:
    review_threshold: float
    deny_threshold: float


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--model-dir", required=True, help="Model dir to write policy.json")

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--scores-f32le",
        default="",
        help="Scores file: f32 little-endian, 1 score per row",
    )
    g.add_argument(
        "--scores-npy", default="", help="Scores file: numpy .npy (float32/float64)"
    )
    g.add_argument(
        "--scores-jsonl",
        default="",
        help="Scores file: jsonl, each line has a score field",
    )

    ap.add_argument(
        "--jsonl-key", default="score", help="Key name used in --scores-jsonl"
    )

    ap.add_argument(
        "--labels-u8",
        default="",
        help=(
            "Optional labels file: uint8 0/1, 1 label per row aligned with scores. "
            "If provided, script prints recall/precision and can enforce --min-recall."
        ),
    )
    ap.add_argument(
        "--min-recall",
        type=float,
        default=None,
        help=(
            "If set (and labels provided), enforce recall >= min_recall for 'flagged' "
            "(score >= review_threshold). May increase actual route_frac."
        ),
    )
    ap.add_argument(
        "--pos-label",
        type=int,
        default=1,
        help="Positive label value in --labels-u8 (default=1).",
    )

    ap.add_argument(
        "--route-frac",
        type=float,
        required=True,
        help="Target fraction routed to NEXT layer (review band). Example: 0.10 means ~10% go to L2.",
    )
    ap.add_argument(
        "--deny-frac",
        type=float,
        required=True,
        help="Target fraction hard-denied at THIS layer. Example: 0.001 means top 0.1%.",
    )

    ap.add_argument(
        "--layer", default="", help="Optional label for metadata (L1/L2/L3)"
    )

    ap.add_argument(
        "--out",
        default="",
        help="Override output path for policy.json (default: model-dir/policy.json)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write files, only print thresholds",
    )

    ap.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="Ensure review_th <= deny_th - eps to avoid empty/invalid bands.",
    )

    return ap.parse_args()


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def load_scores(args: argparse.Namespace):
    try:
        import numpy as np  # type: ignore
    except Exception as e:
        die(f"missing numpy: {e}. Install: pip install numpy")

    if args.scores_f32le:
        p = Path(args.scores_f32le).expanduser().resolve()
        if not p.exists():
            die(f"scores file not found: {p}")
        scores = np.fromfile(p, dtype=np.float32)
        return scores

    if args.scores_npy:
        p = Path(args.scores_npy).expanduser().resolve()
        if not p.exists():
            die(f"scores file not found: {p}")
        scores = np.load(p)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        return scores

    if args.scores_jsonl:
        p = Path(args.scores_jsonl).expanduser().resolve()
        if not p.exists():
            die(f"scores file not found: {p}")
        out = []
        key = str(args.jsonl_key)
        with p.open("r", encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as e:
                    die(f"jsonl parse error at line {ln}: {e}")
                if key not in obj:
                    die(f"jsonl missing key '{key}' at line {ln}")
                out.append(float(obj[key]))
        if not out:
            die("jsonl empty: no scores")
        scores = np.asarray(out, dtype=np.float32)
        return scores

    die("no scores input provided")


def read_labels_u8(path: str, n: int):
    import numpy as np  # type: ignore

    if not path:
        return np.array([], dtype=np.uint8)
    p = Path(path).expanduser().resolve()
    if not p.exists():
        die(f"labels-u8 not found: {p}")
    b = p.read_bytes()
    arr = np.frombuffer(b, dtype=np.uint8)
    if arr.size != n:
        die(f"labels-u8 len mismatch: got={arr.size} expect={n} path={p}")
    return arr


def _rank_threshold_for_min_recall(pos_scores, min_recall: float) -> float:
    """Pick a threshold that guarantees recall >= min_recall under rule: flagged = score >= th.

    We avoid np.quantile interpolation issues by using a rank-based threshold.

    With npos positives, allowing at most fn_max false negatives:
      fn_max = floor((1 - min_recall) * npos)
    Choose th = k-th smallest positive score where k = fn_max (0-based).
    Then exactly fn_max positives are < th (worst case), so recall >= 1 - fn_max/npos.
    """
    import numpy as np  # type: ignore

    pos_scores = np.asarray(pos_scores, dtype=np.float64)
    if pos_scores.size == 0:
        die("no positive scores")
    if not (0.0 < min_recall <= 1.0):
        die("--min-recall must be in (0,1]")

    sp = np.sort(pos_scores)
    npos = int(sp.size)
    fn_max = int(np.floor((1.0 - float(min_recall)) * npos))
    if fn_max < 0:
        fn_max = 0
    if fn_max >= npos:
        fn_max = npos - 1

    th = float(sp[fn_max])
    return th


def compute_policy(
    scores,
    route_frac: float,
    deny_frac: float,
    eps: float,
    labels=None,
    min_recall=None,
    pos_label: int = 1,
):
    """Returns: (final_policy, base_policy, diag)

    Base policy:
      - thresholds from global quantiles to match route_frac/deny_frac.

    Final policy:
      - base policy, optionally adjusted to satisfy label-aware recall constraint.
    """
    import numpy as np  # type: ignore

    route_frac = clamp01(float(route_frac))
    deny_frac = clamp01(float(deny_frac))

    if deny_frac <= 0.0:
        die("--deny-frac must be > 0")
    if route_frac < 0.0:
        die("--route-frac must be >= 0")
    if route_frac + deny_frac >= 1.0:
        die(f"route_frac + deny_frac must be < 1.0 (got {route_frac + deny_frac:.6f})")

    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        die("empty scores")

    # fail-fast for NaN/Inf (they break comparisons and recall accounting)
    if not np.isfinite(scores).all():
        bad = int((~np.isfinite(scores)).sum())
        die(f"scores contain NaN/Inf: bad={bad} / n={scores.size}")

    deny_q = 1.0 - deny_frac
    review_q = 1.0 - (route_frac + deny_frac)

    deny_th = float(np.quantile(scores, deny_q))
    base_review_th = float(np.quantile(scores, review_q))

    base_policy = Policy(
        review_threshold=float(base_review_th), deny_threshold=float(deny_th)
    )
    review_th = base_review_th

    diag = {
        "deny_q": float(deny_q),
        "review_q": float(review_q),
        "base_review_threshold": float(base_review_th),
        "base_deny_threshold": float(deny_th),
        "recall_constraint_applied": False,
        "recall_threshold": None,
        "recall_threshold_method": None,
    }

    if labels is not None and getattr(labels, "size", 0) > 0 and min_recall is not None:
        min_recall = float(min_recall)
        labels = np.asarray(labels)
        if labels.size != scores.size:
            die(f"labels size mismatch: got={labels.size} expect={scores.size}")

        pos_mask = labels == int(pos_label)
        npos = int(pos_mask.sum())
        if npos <= 0:
            die("labels provided but no positive samples found (check --pos-label)")

        pos_scores = scores[pos_mask]
        # rank-based threshold to avoid quantile interpolation surprises
        recall_th = _rank_threshold_for_min_recall(pos_scores, min_recall)

        # enforce recall by lowering review_th if needed
        if base_review_th > recall_th:
            review_th = recall_th
            diag["recall_constraint_applied"] = True
        diag["recall_threshold"] = float(recall_th)
        diag["recall_threshold_method"] = "rank"

    # Ensure ordering (review < deny). Clamp if violated.
    eps = float(eps)
    if eps < 0.0:
        eps = 0.0
    if review_th > (deny_th - eps):
        review_th = deny_th - eps
    if review_th > deny_th:
        review_th = deny_th

    final_policy = Policy(
        review_threshold=float(review_th), deny_threshold=float(deny_th)
    )
    return final_policy, base_policy, diag


def frac_actual(scores, policy: Policy):
    import numpy as np  # type: ignore

    s = scores
    deny = float(np.mean(s >= policy.deny_threshold))
    routed = float(
        np.mean((s >= policy.review_threshold) & (s < policy.deny_threshold))
    )
    allow = float(np.mean(s < policy.review_threshold))
    return {"allow_frac": allow, "route_frac": routed, "deny_frac": deny}


def _stats_with_labels(scores, labels, pos_label: int, policy: Policy):
    import numpy as np  # type: ignore

    pos = labels == int(pos_label)
    flagged = scores >= float(policy.review_threshold)
    denied = scores >= float(policy.deny_threshold)

    tp_flag = int((flagged & pos).sum())
    fp_flag = int((flagged & (~pos)).sum())
    fn_flag = int((~flagged & pos).sum())

    tp_deny = int((denied & pos).sum())
    fp_deny = int((denied & (~pos)).sum())
    fn_deny = int((~denied & pos).sum())

    recall_flag = tp_flag / max(1, (tp_flag + fn_flag))
    prec_flag = tp_flag / max(1, (tp_flag + fp_flag))

    recall_deny = tp_deny / max(1, (tp_deny + fn_deny))
    prec_deny = tp_deny / max(1, (tp_deny + fp_deny))

    return {
        "flagged": {
            "tp": tp_flag,
            "fp": fp_flag,
            "fn": fn_flag,
            "recall": recall_flag,
            "precision": prec_flag,
        },
        "deny": {
            "tp": tp_deny,
            "fp": fp_deny,
            "fn": fn_deny,
            "recall": recall_deny,
            "precision": prec_deny,
        },
    }


def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def main() -> int:
    args = parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.exists():
        die(f"model-dir not found: {model_dir}")

    out_policy = (
        Path(args.out).expanduser().resolve()
        if args.out
        else (model_dir / "policy.json")
    )
    out_meta = model_dir / "policy_calibration.json"

    scores = load_scores(args)
    if scores.size == 0:
        die("empty scores")

    import numpy as np  # type: ignore

    labels = (
        read_labels_u8(args.labels_u8, int(scores.size))
        if args.labels_u8
        else np.array([], dtype=np.uint8)
    )

    final_pol, base_pol, diag = compute_policy(
        scores,
        args.route_frac,
        args.deny_frac,
        float(args.eps),
        labels=(labels if labels.size > 0 else None),
        min_recall=args.min_recall,
        pos_label=int(args.pos_label),
    )

    actual_final = frac_actual(scores, final_pol)
    actual_base = frac_actual(scores, base_pol)

    # Summary
    print("[CALIB] model_dir=", model_dir)
    print("[CALIB] n_scores=", int(scores.size))
    if args.layer:
        print("[CALIB] layer=", args.layer)
    print(
        "[CALIB] target: route_frac={:.6f} deny_frac={:.6f}".format(
            float(args.route_frac), float(args.deny_frac)
        )
    )

    print(
        "[CALIB] base_policy: review_th={:.8f} deny_th={:.8f}".format(
            base_pol.review_threshold, base_pol.deny_threshold
        )
    )
    print(
        "[CALIB] base_actual: allow={:.6f} route={:.6f} deny={:.6f}".format(
            actual_base["allow_frac"],
            actual_base["route_frac"],
            actual_base["deny_frac"],
        )
    )

    print(
        "[CALIB] policy: review_th={:.8f} deny_th={:.8f}".format(
            final_pol.review_threshold, final_pol.deny_threshold
        )
    )
    print(
        "[CALIB] actual: allow={:.6f} route={:.6f} deny={:.6f}".format(
            actual_final["allow_frac"],
            actual_final["route_frac"],
            actual_final["deny_frac"],
        )
    )

    if diag.get("recall_constraint_applied", False):
        print(
            "[CALIB] recall_constraint: min_recall={:.6f} base_review_th={:.8f} recall_th={:.8f} ({})".format(
                float(args.min_recall),
                float(diag.get("base_review_threshold", float("nan"))),
                float(diag.get("recall_threshold", float("nan"))),
                str(diag.get("recall_threshold_method") or "unknown"),
            )
        )

    labels_meta = None
    if labels.size > 0:
        pos = labels == int(args.pos_label)
        npos = int(pos.sum())
        nall = int(labels.size)
        if npos == 0:
            die("labels provided but no positives found (check --pos-label)")

        base_stats = _stats_with_labels(scores, labels, int(args.pos_label), base_pol)
        final_stats = _stats_with_labels(scores, labels, int(args.pos_label), final_pol)

        print(f"[CALIB] labels: n={nall} pos={npos} pos_rate={npos / nall:.6f}")
        print(
            "[CALIB] base_flagged(score>=review_th): tp={} fp={} fn={} recall={:.6f} precision={:.6f}".format(
                base_stats["flagged"]["tp"],
                base_stats["flagged"]["fp"],
                base_stats["flagged"]["fn"],
                base_stats["flagged"]["recall"],
                base_stats["flagged"]["precision"],
            )
        )
        print(
            "[CALIB] base_deny(score>=deny_th): tp={} fp={} fn={} recall={:.6f} precision={:.6f}".format(
                base_stats["deny"]["tp"],
                base_stats["deny"]["fp"],
                base_stats["deny"]["fn"],
                base_stats["deny"]["recall"],
                base_stats["deny"]["precision"],
            )
        )

        print(
            "[CALIB] flagged(score>=review_th): tp={} fp={} fn={} recall={:.6f} precision={:.6f}".format(
                final_stats["flagged"]["tp"],
                final_stats["flagged"]["fp"],
                final_stats["flagged"]["fn"],
                final_stats["flagged"]["recall"],
                final_stats["flagged"]["precision"],
            )
        )
        print(
            "[CALIB] deny(score>=deny_th): tp={} fp={} fn={} recall={:.6f} precision={:.6f}".format(
                final_stats["deny"]["tp"],
                final_stats["deny"]["fp"],
                final_stats["deny"]["fn"],
                final_stats["deny"]["recall"],
                final_stats["deny"]["precision"],
            )
        )

        if args.min_recall is not None and final_stats["flagged"][
            "recall"
        ] + 1e-12 < float(args.min_recall):
            print(
                "[CALIB] WARNING: recall below target: got={:.6f} want>={:.6f}".format(
                    final_stats["flagged"]["recall"], float(args.min_recall)
                )
            )

        if (
            diag.get("recall_constraint_applied", False)
            and actual_final["route_frac"] > float(args.route_frac) + 1e-9
        ):
            print(
                "[CALIB] NOTE: recall constraint can increase route_frac: target={:.6f} actual={:.6f}".format(
                    float(args.route_frac), float(actual_final["route_frac"])
                )
            )

        labels_meta = {
            "labels_u8": str(Path(args.labels_u8).expanduser().resolve())
            if args.labels_u8
            else "",
            "pos_label": int(args.pos_label),
            "n": nall,
            "pos": npos,
            "pos_rate": npos / nall,
            "base": base_stats,
            "final": final_stats,
            "min_recall": (
                float(args.min_recall) if args.min_recall is not None else None
            ),
        }

    if args.dry_run:
        print("[CALIB] dry-run: not writing policy.json")
        return 0

    write_json(
        out_policy,
        {
            "review_threshold": final_pol.review_threshold,
            "deny_threshold": final_pol.deny_threshold,
        },
    )

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "layer": args.layer or "",
        "scores_source": {
            "scores_f32le": args.scores_f32le or "",
            "scores_npy": args.scores_npy or "",
            "scores_jsonl": args.scores_jsonl or "",
            "jsonl_key": args.jsonl_key,
        },
        "n_scores": int(scores.size),
        "target": {
            "route_frac": float(args.route_frac),
            "deny_frac": float(args.deny_frac),
        },
        "diag": diag,
        "labels": labels_meta,
        "base_policy": {
            "review_threshold": base_pol.review_threshold,
            "deny_threshold": base_pol.deny_threshold,
        },
        "base_actual": actual_base,
        "policy": {
            "review_threshold": final_pol.review_threshold,
            "deny_threshold": final_pol.deny_threshold,
        },
        "actual": actual_final,
        "output": {"policy_json": str(out_policy), "calibration_json": str(out_meta)},
    }
    write_json(out_meta, meta)

    print(f"[WRITE] {out_policy}")
    print(f"[WRITE] {out_meta}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
