from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm

from config import load_config
from utils import ensure_dir


def _sample_indices(total: int, target: int) -> List[int]:
    if total <= 0:
        return []
    if total >= target:
        return np.linspace(0, total - 1, target, dtype=int).tolist()
    idx = np.linspace(0, total - 1, total, dtype=int).tolist()
    while len(idx) < target:
        idx.append(idx[-1])
    return idx


def _crop_face(frame_bgr: np.ndarray, box, margin: int, image_size: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if box is None:
        y0 = max(0, h // 2 - image_size // 2)
        x0 = max(0, w // 2 - image_size // 2)
        crop = frame_bgr[y0 : y0 + image_size, x0 : x0 + image_size]
        return cv2.resize(crop, (image_size, image_size))

    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return cv2.resize(frame_bgr, (image_size, image_size))
    return cv2.resize(crop, (image_size, image_size))


def _video_to_sequence(
    video_path: Path,
    mtcnn: MTCNN,
    seq_len: int,
    image_size: int,
    face_margin: int,
) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    indices = _sample_indices(len(frames), seq_len)
    seq = []
    last_box = None
    for idx in indices:
        frame = frames[idx]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        boxes, _ = mtcnn.detect(pil)
        box = boxes[0] if boxes is not None and len(boxes) > 0 else last_box
        face = _crop_face(frame, box, face_margin, image_size)
        if box is not None:
            last_box = box
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32) / 255.0
        seq.append(face.transpose(2, 0, 1))
    return np.stack(seq, axis=0).astype(np.float32)


def preprocess_csv(cfg_path: str, csv_path: str) -> None:
    cfg = load_config(cfg_path)
    paths = cfg["paths"]
    prep = cfg["preprocessing"]

    cache_root = ensure_dir(Path(paths["cache_dir"]))
    split_name = Path(csv_path).stem
    out_dir = ensure_dir(cache_root / split_name)
    out_csv = out_dir / "index.csv"

    df = pd.read_csv(csv_path)
    mtcnn = MTCNN(keep_all=False, device="cuda" if __import__("torch").cuda.is_available() else "cpu")

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Preprocess {split_name}"):
        video_path = Path(row["video_path"])
        vid_id = hashlib.md5(str(video_path).encode("utf-8")).hexdigest()
        npy_path = out_dir / f"{vid_id}.npy"
        if not npy_path.exists():
            arr = _video_to_sequence(
                video_path=video_path,
                mtcnn=mtcnn,
                seq_len=int(prep["sequence_length"]),
                image_size=int(prep["image_size"]),
                face_margin=int(prep["face_margin"]),
            )
            np.save(npy_path, arr)
        rows.append(
            {
                "sequence_path": str(npy_path),
                "label": int(row["label"]),
                "method": row.get("method", "unknown"),
                "video_path": str(video_path),
            }
        )

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved cache index: {out_csv} (rows={len(rows)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess a CSV into cached npy sequences")
    parser.add_argument("--config", required=True)
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    preprocess_csv(args.config, args.csv)
