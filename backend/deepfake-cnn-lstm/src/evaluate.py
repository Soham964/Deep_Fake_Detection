from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import load_config
from dataset import SequenceDataset
from metrics import compute_metrics
from model import CNNLSTM


@torch.no_grad()
def evaluate(cfg_path: str, index_csv: str, checkpoint: str) -> None:
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = SequenceDataset(index_csv)
    loader = DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, num_workers=int(cfg["train"]["num_workers"]), pin_memory=True)

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

    y_true = []
    y_prob = []
    for x, y in tqdm(loader):
        x = x.to(device)
        logits = model(x)
        prob = torch.sigmoid(logits).cpu().numpy()
        y_prob.append(prob)
        y_true.append(y.numpy())

    y_true = np.concatenate(y_true).astype(int)
    y_prob = np.concatenate(y_prob)
    m = compute_metrics(y_true, y_prob, threshold=float(cfg["train"]["threshold"]))
    print(json.dumps(m, indent=2))

    out_dir = Path(cfg["paths"]["checkpoints_dir"])
    out_path = out_dir / f"metrics_{Path(index_csv).stem}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
    print(f"Saved metrics: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on cached index")
    parser.add_argument("--config", required=True)
    parser.add_argument("--index_csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    evaluate(args.config, args.index_csv, args.checkpoint)
