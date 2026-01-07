#!/usr/bin/env python3
"""
把 JSON/JSONL (features object) 转成「raw f32 little-endian dense 向量」的二进制语料。

为什么要这个：
- 你的在线服务如果走 /score_dense_f32（octet-stream），就能彻底干掉 JSON parse / Map lookup / 字符串 key 的开销；
- bench 也就能更接近“纯推理 + 少量 glue”的真实瓶颈。

输出格式（写入 out 文件）：
- 每条样本一个 record，record = dim 个 f32 (little-endian)
- 没有 header（最省）
- dim = feature_names.json 的长度（顺序也必须一致）

用法：
  python3 scripts/jsonl_to_dense_f32le.py \
    --model-dir models/ieee_l1 \
    --jsonl data/bench/ieee_20k.jsonl \
    --out data/bench/ieee_20k.f32 \
    --max 20000

然后压测：
  target/release/risk-bench \
    --url http://127.0.0.1:8080/score_dense_f32 \
    --xgb-dense-file data/bench/ieee_20k.f32 --xgb-dense-dim <DIM> \
    --rps 14000 --duration 20 --concurrency 512
"""

import argparse
import gzip
import json
import os
import struct
import sys
from typing import Any, Dict, List


def load_feature_names(model_dir: str) -> List[str]:
    p = os.path.join(model_dir, "feature_names.json")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_cat_maps(model_dir: str) -> Dict[str, Dict[str, float]]:
    p = os.path.join(model_dir, "cat_maps.json.gz")
    if not os.path.exists(p):
        return {}
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def as_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return str(v)


def build_row(
    obj: Dict[str, Any], feature_names: List[str], cat_maps: Dict[str, Dict[str, float]]
) -> List[float]:
    out: List[float] = []
    for name in feature_names:
        v = obj.get(name, None)
        if isinstance(v, (int, float)):
            out.append(float(v))
            continue

        # categorical mapping
        m = cat_maps.get(name)
        if m is not None:
            s = as_str(v)
            out.append(float(m.get(s, 0.0)))
        else:
            # missing
            out.append(float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument(
        "--jsonl", required=True, help="one-json-object per line; can be plain json too"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--labels-out",
        default="",
        help=(
            "Optional: write aligned labels as raw u8 (0/1), one byte per row. "
            "If present in JSONL, uses top-level key 'label' (preferred) or 'isFraud'."
        ),
    )
    ap.add_argument("--max", type=int, default=20000)
    args = ap.parse_args()

    feature_names = load_feature_names(args.model_dir)
    cat_maps = load_cat_maps(args.model_dir)
    dim = len(feature_names)
    print(
        f"dim={dim}  features={os.path.join(args.model_dir, 'feature_names.json')}",
        file=sys.stderr,
    )

    n = 0
    f_lab = None
    if args.labels_out:
        os.makedirs(os.path.dirname(args.labels_out) or ".", exist_ok=True)
        f_lab = open(args.labels_out, "wb")

    with open(args.jsonl, "r", encoding="utf-8") as fin, open(args.out, "wb") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                v = json.loads(line)
            except Exception:
                # maybe pretty-printed JSON file
                fin.seek(0)
                v = json.load(fin)
                # treat as single sample
                if not isinstance(v, dict):
                    raise SystemExit("json must be an object")
                obj = v.get("features", v)
                if not isinstance(obj, dict):
                    raise SystemExit("json must be an object or have features object")
                row = build_row(obj, feature_names, cat_maps)
                fout.write(struct.pack("<" + "f" * dim, *row))
                n += 1
                break

            if not isinstance(v, dict):
                continue
            obj = v.get("features", v)
            if not isinstance(obj, dict):
                continue
            row = build_row(obj, feature_names, cat_maps)
            fout.write(struct.pack("<" + "f" * dim, *row))

            if f_lab is not None:
                y = v.get("label", v.get("isFraud", 0))
                try:
                    yy = int(y) & 0xFF
                except Exception:
                    yy = 0
                f_lab.write(bytes([yy]))

            n += 1
            if n >= args.max:
                break

    if f_lab is not None:
        f_lab.close()

    print(f"wrote {n} records => {args.out}", file=sys.stderr)
    if args.labels_out:
        print(f"wrote {n} labels  => {args.labels_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
