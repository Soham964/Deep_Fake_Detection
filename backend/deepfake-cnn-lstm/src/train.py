from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import load_config
from dataset import SequenceDataset
from metrics import compute_metrics
from model import CNNLSTM
from utils import ensure_dir, set_seed


def run_epoch(
    model: CNNLSTM,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer=None,
    scaler=None,
    amp: bool = True,
) -> Tuple[float, Dict[str, float]]:
    train_mode = optimizer is not None
    model.train(train_mode)

    losses = []
    all_true = []
    all_prob = []

    for x, y in tqdm(loader, leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            logits = model(x)
            loss = criterion(logits, y)

        if train_mode:
            if scaler is not None and amp and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        losses.append(float(loss.item()))
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_prob.append(probs)
        all_true.append(y.detach().cpu().numpy())

    y_true = np.concatenate(all_true).astype(int)
    y_prob = np.concatenate(all_prob)
    metrics = compute_metrics(y_true, y_prob)
    return float(np.mean(losses)), metrics


def train(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_dir = Path(cfg["paths"]["cache_dir"])
    train_index = cache_dir / "train" / "index.csv"
    val_index = cache_dir / "val" / "index.csv"

    train_ds = SequenceDataset(train_index)
    val_ds = SequenceDataset(val_index)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=True,
    )

    model = CNNLSTM(
        backbone_name=cfg["model"]["backbone"],
        pretrained=bool(cfg["model"]["pretrained"]),
        lstm_hidden=int(cfg["model"]["lstm_hidden"]),
        lstm_layers=int(cfg["model"]["lstm_layers"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    amp = bool(cfg["train"]["amp"])
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")

    ckpt_dir = ensure_dir(cfg["paths"]["checkpoints_dir"])
    best_f1 = -1.0
    patience = int(cfg["train"]["early_stopping_patience"])
    stale = 0

    # Stage A: freeze CNN
    model.freeze_backbone()
    opt = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["train"]["lr_stage_a"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    sched = ReduceLROnPlateau(opt, mode="max", patience=2, factor=0.5)

    total_epochs = int(cfg["train"]["epochs_stage_a"]) + int(cfg["train"]["epochs_stage_b"])
    history = []
    epoch_num = 0

    for _ in range(int(cfg["train"]["epochs_stage_a"])):
        epoch_num += 1
        tr_loss, tr_m = run_epoch(model, train_loader, criterion, device, opt, scaler, amp)
        va_loss, va_m = run_epoch(model, val_loader, criterion, device, None, scaler, amp)
        sched.step(va_m["f1"])
        rec = {"epoch": epoch_num, "stage": "A", "train_loss": tr_loss, "val_loss": va_loss, **{f"train_{k}": v for k, v in tr_m.items()}, **{f"val_{k}": v for k, v in va_m.items()}}
        history.append(rec)
        print(rec)

        if va_m["f1"] > best_f1:
            best_f1 = va_m["f1"]
            stale = 0
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch_num}, ckpt_dir / "best.pt")
        else:
            stale += 1
            if stale >= patience:
                break

    # Stage B: unfreeze CNN and continue
    model.unfreeze_backbone()
    opt = AdamW(model.parameters(), lr=float(cfg["train"]["lr_stage_b"]), weight_decay=float(cfg["train"]["weight_decay"]))
    sched = ReduceLROnPlateau(opt, mode="max", patience=2, factor=0.5)

    for _ in range(int(cfg["train"]["epochs_stage_b"])):
        epoch_num += 1
        if epoch_num > total_epochs:
            break
        tr_loss, tr_m = run_epoch(model, train_loader, criterion, device, opt, scaler, amp)
        va_loss, va_m = run_epoch(model, val_loader, criterion, device, None, scaler, amp)
        sched.step(va_m["f1"])
        rec = {"epoch": epoch_num, "stage": "B", "train_loss": tr_loss, "val_loss": va_loss, **{f"train_{k}": v for k, v in tr_m.items()}, **{f"val_{k}": v for k, v in va_m.items()}}
        history.append(rec)
        print(rec)

        if va_m["f1"] > best_f1:
            best_f1 = va_m["f1"]
            stale = 0
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch_num}, ckpt_dir / "best.pt")
        else:
            stale += 1
            if stale >= patience:
                break

    with open(ckpt_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Training done. Best val F1={best_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN-LSTM deepfake classifier")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(args.config)
