#!/usr/bin/env python3
"""
Compute and cache ImageNet reference statistics for FID/IS (pure PyTorch).
"""

import argparse
import logging
import os

from utils.fid_util import compute_fid_stats
from utils.logging_util import log_for_0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet_root", required=True, type=str)
    parser.add_argument("--output", required=True, type=str, help="Output .npz path")
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--image_size", default=256, type=int)
    parser.add_argument("--batch_size", default=200, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not os.path.exists(args.imagenet_root):
        raise ValueError(f"ImageNet root path does not exist: {args.imagenet_root}")
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    log_for_0("=" * 50)
    log_for_0("COMPUTING FID REFERENCE STATS")
    log_for_0("=" * 50)
    fid_stats_path = compute_fid_stats(
        imagenet_root=args.imagenet_root,
        split=args.split,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_path=args.output,
    )
    log_for_0(f"FID statistics saved to: {fid_stats_path}")
    log_for_0("=" * 50)
    log_for_0("DONE")
    log_for_0("=" * 50)


if __name__ == "__main__":
    main()
