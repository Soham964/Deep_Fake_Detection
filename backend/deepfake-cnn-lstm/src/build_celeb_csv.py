from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from config import load_config
from utils import ensure_dir


def list_videos(folder: Path, extensions: Iterable[str]):
    ext_set = {e.lower() for e in extensions}
    return [p for p in sorted(folder.rglob("*")) if p.is_file() and p.suffix.lower() in ext_set]


def build_celeb_csv(cfg_path: str, celeb_root: str, real_dir_name: str = "real", fake_dir_name: str = "fake") -> None:
    cfg = load_config(cfg_path)
    out_csv = Path(cfg["paths"]["celeb_test_csv"])
    ensure_dir(out_csv.parent)

    root = Path(celeb_root)
    exts = cfg["data"]["video_extensions"]
    real_videos = list_videos(root / real_dir_name, exts)
    fake_videos = list_videos(root / fake_dir_name, exts)

    rows = []
    for p in real_videos:
        rows.append({"video_path": str(p), "label": 0, "method": "celeb_real"})
    for p in fake_videos:
        rows.append({"video_path": str(p), "label": 1, "method": "celeb_fake"})

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv} (rows={len(df)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Celeb-DF test CSV")
    parser.add_argument("--config", required=True)
    parser.add_argument("--celeb_root", required=True)
    parser.add_argument("--real_dir", default="real")
    parser.add_argument("--fake_dir", default="fake")
    args = parser.parse_args()
    build_celeb_csv(args.config, args.celeb_root, args.real_dir, args.fake_dir)
