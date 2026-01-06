#!/usr/bin/env python3
"""
Export ieee_l1 XGBoost model to C code using TL2cgen (Treelite successor).

This script generates a C source package that exposes:
  - size_t get_num_feature(void);
  - float predict(Entry* data, int pred_margin);

See TL2cgen docs "Deploying models" for the C API.

Example:
  python3 scripts/ml/export_l1_tl2cgen_c.py \
      --model models/ieee_l1/ieee_xgb.ubj \
      --out crates/risk-core/native/tl2cgen/ieee_l1 \
      --quantize 1 \
      --parallel-comp 8

Then compile the server with:
  cargo build -p risk-server-glommio --release --no-default-features --features native_l1_tl2cgen

Notes:
- `--quantize 1` typically yields faster inference and smaller code, at the cost of slight numerical differences.
- `--parallel-comp N` splits the generated code into N translation units to speed up C compilation.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to XGBoost model (.ubj / .json)")
    ap.add_argument("--out", required=True, help="Output directory for generated C code")
    ap.add_argument("--quantize", type=int, default=1, choices=[0, 1], help="Enable quantized codegen")
    ap.add_argument(
        "--parallel-comp",
        type=int,
        default=8,
        help="Number of translation units (0 = single TU). Speeds up C compilation for huge models.",
    )
    args = ap.parse_args()

    try:
        import treelite  # type: ignore
        import treelite.frontend  # type: ignore
        import tl2cgen  # type: ignore
    except Exception as e:
        print(
            "ERROR: missing python deps. Install: pip install treelite tl2cgen\n"
            f"Import error: {e}",
            file=sys.stderr,
        )
        return 2

    model_path = Path(args.model).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        return 2

    # Clean output dir (to avoid stale .c files)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    try:
        tl_model = treelite.frontend.load_xgboost_model(
            str(model_path), format_choice="use_suffix"
        )
    except Exception:
        # Fallback: explicit model_format may be needed for some versions.
        tl_model = treelite.Model.load(str(model_path))

    params = {
        "quantize": int(args.quantize),
        "parallel_comp": int(args.parallel_comp),
    }

    print(f"[tl2cgen] loading model: {model_path}")
    print(f"[tl2cgen] generating C code into: {out_dir}")
    print(f"[tl2cgen] params: {params}")

    tl2cgen.generate_c_code(tl_model, dirpath=str(out_dir), params=params)

    # A sanity check: ensure we got at least one .c file.
    c_files = list(out_dir.glob("*.c"))
    h_files = list(out_dir.glob("*.h"))
    if not c_files or not h_files:
        print(f"ERROR: output missing .c/.h files: {out_dir}", file=sys.stderr)
        return 2

    print(f"[tl2cgen] done: {len(c_files)} C files, {len(h_files)} headers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
