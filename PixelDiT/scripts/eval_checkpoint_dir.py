#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _default_output_root(run_dir: Path) -> Path:
    return run_dir / "batch_eval" / _now_stamp()


def _is_checkpoint_candidate(path: Path) -> bool:
    if path.suffix != ".pt":
        return False
    banned_parts = {"eval", "tensorboard", "tracebacks", "batch_eval"}
    if any(part in banned_parts for part in path.parts):
        return False
    return True


def _discover_checkpoints(run_dir: Path) -> List[Path]:
    checkpoints: List[Path] = []
    for path in run_dir.rglob("*.pt"):
        if _is_checkpoint_candidate(path):
            checkpoints.append(path.resolve())
    seen = set()
    deduped: List[Path] = []
    for path in sorted(checkpoints):
        if path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


def _sanitize_label(path: Path, run_dir: Path) -> str:
    rel = str(path.resolve().relative_to(run_dir.resolve()))
    label = re.sub(r"[^A-Za-z0-9._-]+", "_", rel)
    if label.endswith(".pt"):
        label = label[:-3]
    return label


def _find_summary_json(workdir: Path) -> Optional[Path]:
    candidates = sorted((workdir / "eval").glob("*/metrics_summary.json"))
    if not candidates:
        return None
    return candidates[-1]


def _score_value(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return float("-inf")
    if math.isnan(out):
        return float("-inf")
    return out


def _best_entry(entries: Dict[str, dict], score_key: str) -> Tuple[Optional[str], Optional[dict]]:
    best_tag = None
    best_payload = None
    best_score = float("-inf")
    for tag, payload in entries.items():
        score = _score_value(payload.get(score_key, float("-inf")))
        if score > best_score:
            best_score = score
            best_tag = tag
            best_payload = payload
    return best_tag, best_payload


def _flatten_checkpoint_summary(summary: dict, checkpoint_label: str, checkpoint_path: str, eval_dir: str) -> dict:
    row = {
        "checkpoint_label": checkpoint_label,
        "checkpoint_path": checkpoint_path,
        "eval_dir": eval_dir,
        "config": summary.get("config", ""),
        "global_step": summary.get("global_step", ""),
        "epoch": summary.get("epoch", ""),
    }

    posthoc_tag, posthoc_payload = _best_entry(summary.get("posthoc", {}), "score_posthoc")
    regen_tag, regen_payload = _best_entry(summary.get("regen", {}), "score_regen")

    row["best_posthoc_tag"] = posthoc_tag or ""
    row["best_regen_tag"] = regen_tag or ""

    if posthoc_payload is not None:
        mean = posthoc_payload.get("mean_over_t", {})
        row["best_posthoc_score"] = posthoc_payload.get("score_posthoc", "")
        row["best_posthoc_psnr"] = mean.get("psnr", "")
        row["best_posthoc_ssim"] = mean.get("ssim", "")
        row["best_posthoc_lpips"] = mean.get("lpips", "")
        sweep = posthoc_payload.get("sweep", {})
        row["best_posthoc_num_steps"] = sweep.get("num_steps", "")
        row["best_posthoc_omega"] = sweep.get("omega", "")
        row["best_posthoc_t_min"] = sweep.get("t_min", "")
        row["best_posthoc_t_max"] = sweep.get("t_max", "")

    if regen_payload is not None:
        mean = regen_payload.get("mean_over_t", {})
        row["best_regen_score"] = regen_payload.get("score_regen", "")
        row["best_regen_dino_sim"] = mean.get("dino_sim", "")
        row["best_regen_clip_sim"] = mean.get("clip_sim", "")
        row["best_regen_div_lpips"] = mean.get("div_lpips", "")
        row["best_regen_pareto"] = mean.get("pareto", "")
        sweep = regen_payload.get("sweep", {})
        row["best_regen_num_steps"] = sweep.get("num_steps", "")
        row["best_regen_omega"] = sweep.get("omega", "")
        row["best_regen_t_min"] = sweep.get("t_min", "")
        row["best_regen_t_max"] = sweep.get("t_max", "")

    return row


def _flatten_sweeps(summary: dict, checkpoint_label: str, checkpoint_path: str) -> List[dict]:
    rows: List[dict] = []
    base = {
        "checkpoint_label": checkpoint_label,
        "checkpoint_path": checkpoint_path,
        "global_step": summary.get("global_step", ""),
        "epoch": summary.get("epoch", ""),
    }
    for family, score_key in (("posthoc", "score_posthoc"), ("regen", "score_regen")):
        for tag, payload in summary.get(family, {}).items():
            row = dict(base)
            row["family"] = family
            row["sweep_tag"] = tag
            row["score"] = payload.get(score_key, "")
            sweep = payload.get("sweep", {})
            row["num_steps"] = sweep.get("num_steps", "")
            row["omega"] = sweep.get("omega", "")
            row["t_min"] = sweep.get("t_min", "")
            row["t_max"] = sweep.get("t_max", "")
            for key, value in payload.get("mean_over_t", {}).items():
                row[f"mean_{key}"] = value
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_eval_subprocess(
    repo_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    workdir: Path,
    device: str,
    max_samples: Optional[int],
    num_workers_override: Optional[int],
) -> int:
    cmd = [
        sys.executable,
        str((repo_root / "PixelDiT" / "eval.py").resolve()),
        "--config",
        str(config_path.resolve()),
        "--checkpoint",
        str(checkpoint_path.resolve()),
        "--workdir",
        str(workdir.resolve()),
        "--device",
        str(device),
    ]
    if max_samples is not None:
        cmd.extend(["--max-samples", str(int(max_samples))])
    if num_workers_override is not None:
        cmd.extend(["--num-workers-override", str(int(num_workers_override))])
    env = os.environ.copy()
    env.setdefault("PYTHONFAULTHANDLER", "1")
    return subprocess.run(cmd, cwd=str(repo_root), env=env, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-evaluate all checkpoints in a PixelDiT run directory.")
    parser.add_argument("--run-dir", required=True, type=str, help="Run directory containing used_config.yml/checkpoints.")
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Optional eval config path. Defaults to <run-dir>/used_config.yml.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="",
        help="Root directory for batch-eval outputs. Defaults to <run-dir>/batch_eval/<timestamp>.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers-override", type=int, default=None)
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failed checkpoint eval.")
    parser.add_argument("--dry-run", action="store_true", help="Only discover checkpoints and planned output paths.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run dir not found: {run_dir}")

    config_path = Path(args.config).expanduser().resolve() if args.config.strip() else (run_dir / "used_config.yml").resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")

    checkpoints = _discover_checkpoints(run_dir)
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint .pt files found under: {run_dir}")

    output_root = Path(args.output_root).expanduser().resolve() if args.output_root.strip() else _default_output_root(run_dir)
    print(f"[batch-eval] run_dir: {run_dir}")
    print(f"[batch-eval] config: {config_path}")
    print(f"[batch-eval] checkpoints_found: {len(checkpoints)}")
    for ckpt in checkpoints:
        print(f"[batch-eval] checkpoint: {ckpt}")

    if args.dry_run:
        return 0

    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_dir": str(run_dir),
        "config": str(config_path),
        "output_root": str(output_root),
        "device": args.device,
        "max_samples": args.max_samples,
        "num_workers_override": args.num_workers_override,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "items": [],
    }

    checkpoint_rows: List[dict] = []
    sweep_rows: List[dict] = []

    for idx, checkpoint_path in enumerate(checkpoints, start=1):
        label = _sanitize_label(checkpoint_path, run_dir)
        ckpt_workdir = output_root / label
        ckpt_workdir.mkdir(parents=True, exist_ok=True)
        print(f"[batch-eval] ({idx}/{len(checkpoints)}) evaluating {label}")
        returncode = _run_eval_subprocess(
            repo_root=repo_root,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            workdir=ckpt_workdir,
            device=args.device,
            max_samples=args.max_samples,
            num_workers_override=args.num_workers_override,
        )

        item = {
            "checkpoint_label": label,
            "checkpoint_path": str(checkpoint_path),
            "workdir": str(ckpt_workdir),
            "returncode": int(returncode),
        }

        summary_path = _find_summary_json(ckpt_workdir)
        if summary_path is not None and summary_path.is_file():
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            item["summary_json"] = str(summary_path)
            item["global_step"] = summary.get("global_step", "")
            item["epoch"] = summary.get("epoch", "")
            checkpoint_rows.append(
                _flatten_checkpoint_summary(
                    summary=summary,
                    checkpoint_label=label,
                    checkpoint_path=str(checkpoint_path),
                    eval_dir=str(summary_path.parent),
                )
            )
            sweep_rows.extend(_flatten_sweeps(summary, checkpoint_label=label, checkpoint_path=str(checkpoint_path)))
        else:
            item["summary_json"] = ""

        manifest["items"].append(item)

        if returncode != 0 and args.fail_fast:
            break

    manifest_path = output_root / "batch_eval_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    checkpoint_csv = output_root / "checkpoint_summary.csv"
    sweep_csv = output_root / "sweep_summary.csv"
    _write_csv(checkpoint_csv, checkpoint_rows)
    _write_csv(sweep_csv, sweep_rows)

    print(f"[batch-eval] manifest: {manifest_path}")
    if checkpoint_rows:
        print(f"[batch-eval] checkpoint summary csv: {checkpoint_csv}")
    if sweep_rows:
        print(f"[batch-eval] sweep summary csv: {sweep_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
