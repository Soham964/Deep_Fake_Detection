from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from config import load_config
from utils import ensure_dir, set_seed


def list_videos(folder: Path, extensions: Iterable[str]) -> List[Path]:
    ext_set = {e.lower() for e in extensions}
    return [p for p in sorted(folder.rglob("*")) if p.is_file() and p.suffix.lower() in ext_set]


def sample_fake_per_folder(folder: Path, k: int, extensions: Iterable[str], seed: int) -> List[Path]:
    videos = list_videos(folder, extensions)
    if len(videos) < k:
        raise ValueError(f"Folder {folder} has only {len(videos)} videos, requested {k}")
    rng = random.Random(seed)
    rng.shuffle(videos)
    return sorted(videos[:k])


def build_metadata(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    set_seed(cfg["seed"])

    dataset_root = Path(cfg["paths"]["dataset_root"])
    out_csv = Path(cfg["paths"]["metadata_csv"])
    ensure_dir(out_csv.parent)

    extensions = cfg["data"]["video_extensions"]
    real_folder = dataset_root / cfg["data"]["real_folder"]
    fake_folders = [dataset_root / f for f in cfg["data"]["fake_folders"]]
    fake_per_folder = int(cfg["data"]["fake_per_folder"])
    seed = int(cfg["seed"])

    real_videos = list_videos(real_folder, extensions)
    rows = []
    for p in real_videos:
        rows.append({"video_path": str(p), "label": 0, "method": "original"})

    for idx, folder in enumerate(fake_folders):
        sampled = sample_fake_per_folder(folder, fake_per_folder, extensions, seed + idx + 1)
        for p in sampled:
            rows.append({"video_path": str(p), "label": 1, "method": folder.name})

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved metadata: {out_csv} (rows={len(df)})")


def stratified_split(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    seed = int(cfg["seed"])

    metadata_csv = Path(cfg["paths"]["metadata_csv"])
    train_csv = Path(cfg["paths"]["train_csv"])
    val_csv = Path(cfg["paths"]["val_csv"])
    ensure_dir(train_csv.parent)

    train_ratio = float(cfg["data"]["train_ratio"])

    df = pd.read_csv(metadata_csv)
    train_parts = []
    val_parts = []
    for label, group in df.groupby("label"):
        group = group.sample(frac=1.0, random_state=seed + int(label)).reset_index(drop=True)
        cut = int(len(group) * train_ratio)
        train_parts.append(group.iloc[:cut])
        val_parts.append(group.iloc[cut:])

    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    print(f"Saved train: {train_csv} (rows={len(train_df)})")
    print(f"Saved val: {val_csv} (rows={len(val_df)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build metadata and train/val split")
    parser.add_argument("--config", required=True, help="Path to config yaml")
    parser.add_argument("--step", choices=["metadata", "split", "all"], default="all")
    args = parser.parse_args()

    if args.step in ("metadata", "all"):
        build_metadata(args.config)
    if args.step in ("split", "all"):
        stratified_split(args.config)
