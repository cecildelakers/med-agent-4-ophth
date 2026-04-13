#!/usr/bin/env python3
"""
Download Qwen/Qwen3-4B to a local directory with resume support.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Qwen3-4B model files.")
    parser.add_argument(
        "--repo-id",
        default="Qwen/Qwen3-4B",
        help="Hugging Face model repo id.",
    )
    parser.add_argument(
        "--output-dir",
        default="/root/autodl-tmp/models/Qwen3-4B",
        help="Local model folder.",
    )
    parser.add_argument(
        "--hf-endpoint",
        default=os.getenv("HF_ENDPOINT", ""),
        help="Optional mirror endpoint, e.g. https://hf-mirror.com",
    )
    parser.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN", ""),
        help="Optional HF token for gated models.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    from huggingface_hub import snapshot_download  # lazy import

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        token=args.hf_token if args.hf_token else None,
    )

    print(f"download_complete: {args.repo_id} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

