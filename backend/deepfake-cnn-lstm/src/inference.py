from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch
from facenet_pytorch import MTCNN

from config import load_config
from model import CNNLSTM
from preprocess import _video_to_sequence


@torch.no_grad()
def predict_video(cfg_path: str, checkpoint: str, video_path: str) -> None:
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNLSTM(
        backbone_name=cfg["model"]["backbone"],
        pretrained=False,
        lstm_hidden=int(cfg["model"]["lstm_hidden"]),
        lstm_layers=int(cfg["model"]["lstm_layers"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    mtcnn = MTCNN(keep_all=False, device=device.type)
    arr = _video_to_sequence(
        Path(video_path),
        mtcnn=mtcnn,
        seq_len=int(cfg["preprocessing"]["sequence_length"]),
        image_size=int(cfg["preprocessing"]["image_size"]),
        face_margin=int(cfg["preprocessing"]["face_margin"]),
    )
    x = torch.from_numpy(arr).unsqueeze(0).to(device)
    prob_fake = torch.sigmoid(model(x)).item()
    thr = float(cfg["train"]["threshold"])
    label = "fake" if prob_fake >= thr else "real"
    print(f"video={video_path}")
    print(f"prediction={label}")
    print(f"confidence_fake={prob_fake:.4f}")
    print(f"threshold={thr}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict real/fake for one video")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video", required=True)
    args = parser.parse_args()
    predict_video(args.config, args.checkpoint, args.video)
