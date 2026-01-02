import argparse
import gzip
import json
from pathlib import Path


def select_cols(all_cols):
    # v0：先选一批“强且常见”的列，避免一上来全量把内存炸穿
    keep = set(["TransactionAmt", "ProductCD", "P_emaildomain", "R_emaildomain"])
    out = []
    for c in all_cols:
        if c in ("isFraud", "TransactionID", "TransactionDT"):
            continue
        if (
            c in keep
            or c.startswith("card")
            or c.startswith("addr")
            or c.startswith("dist")
            or c.startswith("C")
            or c.startswith("D")
            or c.startswith("M")
            or c.startswith("id_")
            or c.startswith("Device")
        ):
            out.append(c)
    return out


def build_freq_mapping(train: pl.DataFrame, col: str, topk: int):
    vc = (
        train.select(pl.col(col).cast(pl.Utf8).fill_null("__NULL__"))
        .group_by(col)
        .len()
        .sort("len", descending=True)
    )
    if topk > 0:
        vc = vc.head(topk)

    n = train.height
    mapping = vc.with_columns((pl.col("len") / pl.lit(n)).alias(f"{col}__freq")).select(
        pl.col(col), pl.col(f"{col}__freq")
    )
    mp = {str(k): float(v) for k, v in mapping.iter_rows()}
    return mapping, mp


def apply_freq_mapping(df: pl.DataFrame, col: str, mapping: pl.DataFrame):
    return (
        df.with_columns(pl.col(col).cast(pl.Utf8).fill_null("__NULL__"))
        .join(mapping, on=col, how="left")
        .with_columns(pl.col(f"{col}__freq").fill_null(0.0).cast(pl.Float32))
        .drop(col)
        .rename({f"{col}__freq": col})
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-tx", required=True)
    ap.add_argument("--train-id", required=True)
    ap.add_argument("--out-dir", default="models/ieee_cis_xgb")
    ap.add_argument("--valid-frac", type=float, default=0.2)
    ap.add_argument("--cat-topk", type=int, default=2000)
    ap.add_argument("--limit-rows", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample-bytree", type=float, default=0.8)
    ap.add_argument("--max-bin", type=int, default=256)
    ap.add_argument("--num-round", type=int, default=5000)
    ap.add_argument("--early-stop", type=int, default=200)

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) lazy scan + join（只在最后 collect）
    tx = pl.scan_csv(args.train_tx, ignore_errors=True)
    idn = pl.scan_csv(args.train_id, ignore_errors=True)

    lf = tx.join(idn, on="TransactionID", how="left")

    if args.limit_rows and args.limit_rows > 0:
        lf = lf.limit(args.limit_rows)

    # 2) 先拿 schema 决定特征列
    schema = lf.collect_schema()
    cols = list(schema.keys())

    if "isFraud" not in cols or "TransactionDT" not in cols:
        raise RuntimeError("missing isFraud/TransactionDT (train files?)")

    feat_cols = select_cols(cols)
    if not feat_cols:
        # fallback：除了 id/label/time 之外全选（不推荐，但至少能跑）
        feat_cols = [
            c for c in cols if c not in ("TransactionID", "TransactionDT", "isFraud")
        ]

    use_cols = ["TransactionID", "TransactionDT", "isFraud"] + feat_cols

    df = lf.select([pl.col(c) for c in use_cols]).collect()

    cut = df.select(
        pl.col("TransactionDT").quantile(1.0 - args.valid_frac, "nearest")
    ).item()
    train = df.filter(pl.col("TransactionDT") <= pl.lit(cut))
    valid = df.filter(pl.col("TransactionDT") > pl.lit(cut))

    # 4) 识别类别列（Utf8 / Categorical），先统一 cast
    #    注意：IEEE-CIS 很多列是字符串/混合类型，cast 会把非数值变 null
    cat_cols = []
    num_cols = []
    for c in feat_cols:
        dtp = train.schema.get(c)
        if dtp in (pl.Utf8, pl.Categorical):
            cat_cols.append(c)
        else:
            num_cols.append(c)

    # 5) 数值列：cast float32，null -> NaN
    def prep_num(d: pl.DataFrame):
        out = d.select(
            [pl.col("isFraud")]
            + [pl.col(c).cast(pl.Float32, strict=False) for c in num_cols]
            + [pl.col(c) for c in cat_cols]
        )
        # polars 的 null 最后转 numpy 会变 nan（float列），这正好给 xgb 用
        return out

    train = prep_num(train)
    valid = prep_num(valid)

    # 6) 类别列：freq encoding（只用 train 统计），映射字典导出
    cat_maps = {}
    for c in cat_cols:
        mapping_df, mp = build_freq_mapping(train, c, args.cat_topk)
        train = apply_freq_mapping(train, c, mapping_df)
        valid = apply_freq_mapping(valid, c, mapping_df)
        cat_maps[c] = mp

    feature_names = num_cols + cat_cols  # 现在 cat_cols 已被替换成频率列（同名）

    # 7) 转 numpy 喂给 xgboost（最后一公里）
    y_train = train.select("isFraud").to_numpy().reshape(-1).astype(np.float32)
    y_valid = valid.select("isFraud").to_numpy().reshape(-1).astype(np.float32)

    X_train = train.select(feature_names).to_numpy()
    X_valid = valid.select(feature_names).to_numpy()

    # 类别不平衡权重
    pos = float((y_train > 0.5).sum())
    neg = float(len(y_train)) - pos
    spw = neg / max(1.0, pos)

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "aucpr"],
        "tree_method": "hist",
        "max_depth": args.max_depth,
        "eta": args.eta,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "max_bin": args.max_bin,
        "seed": args.seed,
        "scale_pos_weight": spw,
    }

    dtrain = xgb.QuantileDMatrix(
        X_train, label=y_train, feature_names=feature_names, max_bin=args.max_bin
    )
    dvalid = xgb.QuantileDMatrix(
        X_valid,
        label=y_valid,
        feature_names=feature_names,
        max_bin=args.max_bin,
        ref=dtrain,
    )

    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.num_round,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=args.early_stop,
        verbose_eval=50,
    )

    p = bst.predict(dvalid)
    auc = roc_auc_score(y_valid, p)
    prauc = average_precision_score(y_valid, p)

    # 8) 导出产物
    (out_dir / "xgb_model.json").write_text(bst.save_raw("json").decode("utf-8"))
    (out_dir / "feature_names.json").write_text(
        json.dumps(feature_names, ensure_ascii=False, indent=2)
    )

    with gzip.open(out_dir / "cat_maps.json.gz", "wt", encoding="utf-8") as f:
        json.dump(cat_maps, f, ensure_ascii=False)

    metrics = {
        "auc": float(auc),
        "prauc": float(prauc),
        "best_iteration": int(bst.best_iteration),
        "train_rows": int(len(y_train)),
        "valid_rows": int(len(y_valid)),
        "n_features": int(len(feature_names)),
        "scale_pos_weight": float(spw),
        "cut_dt": int(cut),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2)
    )

    # 预测时建议显式用 best_iteration（避免 early stopping 后还用全树）
    best_it = getattr(bst, "best_iteration", None)
    if best_it is not None:
        p_valid = bst.predict(dvalid, iteration_range=(0, best_it + 1))
    else:
        p_valid = bst.predict(dvalid)

    # ===== 1) Top20(gain) =====
    imp = bst.get_score(importance_type="gain")
    top20 = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:20]

    (out_dir / "importance_top20.json").write_text(
        json.dumps(
            [{"feature": k, "gain": float(v)} for k, v in top20],
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\nTop20(gain):")
    for k, v in top20:
        print(k, v)

    # ===== 2) 阈值点（ROC / PR） =====
    y_true = y_valid  # 你的 y_valid 已经是 numpy float32 了
    fpr, tpr, thr = roc_curve(y_true, p_valid)

    def pick_nearest(target_arr, target):
        i = int(np.argmin(np.abs(target_arr - target)))
        return i

    roc_points = []
    for target_fpr in [0.001, 0.005, 0.01]:  # 0.1%, 0.5%, 1%
        i = pick_nearest(fpr, target_fpr)
        roc_points.append(
            {
                "target_fpr": float(target_fpr),
                "threshold": float(thr[i]),
                "fpr": float(fpr[i]),
                "tpr": float(tpr[i]),
            }
        )

    prec, rec, thr2 = precision_recall_curve(y_true, p_valid)
    pr_points = []
    for target_p in [0.90, 0.95]:
        j = pick_nearest(prec, target_p)
        # thr2 比 prec/rec 少一个元素：阈值对应的是 thr2[j-1]
        t = float(thr2[max(0, j - 1)]) if len(thr2) > 0 else 0.5
        pr_points.append(
            {
                "target_precision": float(target_p),
                "threshold": t,
                "precision": float(prec[j]),
                "recall": float(rec[j]),
            }
        )

    (out_dir / "threshold_points.json").write_text(
        json.dumps(
            {"roc_points": roc_points, "pr_points": pr_points},
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\nThreshold points saved:", str(out_dir / "threshold_points.json"))
    print("Importance saved:", str(out_dir / "importance_top20.json"))
    print("DONE:", json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
