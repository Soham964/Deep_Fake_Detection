"""
finetune.py — Fine-tune ResNeXt on feedback data from feedback.db
Run: python finetune.py
"""

import os, sys, sqlite3
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize
from torchvision import transforms as T
from PIL import Image
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "feedback.db")
DFDC_DIR = os.path.join(BASE_DIR, "kaggle-dfdc-master", "kaggle-dfdc-master")

sys.path.insert(0, DFDC_DIR)
sys.path.insert(0, os.path.join(DFDC_DIR, "external", "Pytorch_Retinaface"))

# ── Config ─────────────────────────────────────────────────
EPOCHS     = 5
LR         = 1e-5      # small LR for fine-tuning
BATCH_SIZE = 4
FACE_LIMIT = 16        # faces per video (less = faster)
FRAME_SKIP = 9

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load face detector ─────────────────────────────────────
from face_utils import norm_crop, FaceDetector
retinaface = FaceDetector(device=device.type)
retinaface.load_checkpoint(os.path.join(DFDC_DIR, "pth_fiels", "RetinaFace-Resnet50-fixed.pth"))

mean      = [0.485, 0.456, 0.406]
std       = [0.229, 0.224, 0.225]
normalize = Normalize(mean, std)
to_tensor = T.ToTensor()

# ── Extract faces from video ───────────────────────────────
def extract_faces(video_path, face_limit=FACE_LIMIT):
    cap   = cv2.VideoCapture(video_path)
    faces = []
    while len(faces) < face_limit:
        for _ in range(FRAME_SKIP):
            cap.grab()
        ok, img = cap.read()
        if not ok:
            break
        boxes, landms = retinaface.detect(img)
        if boxes.shape[0] == 0:
            continue
        areas = (boxes[:,3]-boxes[:,1]) * (boxes[:,2]-boxes[:,0])
        idx   = areas.argmax()
        lm    = landms[idx].detach().numpy().reshape(5,2).astype(int)
        crop  = norm_crop(img, lm, image_size=224)
        face  = Image.fromarray(crop[:,:,::-1])
        faces.append(normalize(to_tensor(face)))
    cap.release()
    return faces

# ── Load data from DB ──────────────────────────────────────
con  = sqlite3.connect(DB_PATH)
rows = con.execute(
    "SELECT video_path, true_label FROM feedback WHERE video_path IS NOT NULL"
).fetchall()
con.close()

# Filter to only existing files upfront
rows = [(p, l) for p, l in rows if os.path.exists(p)]
print(f"\nFound {len(rows)} existing labeled videos...")

samples = []   # list of (tensor, label)
skipped = 0

for path, label in rows:
    if not os.path.exists(path):
        print(f"  SKIP (not found): {path}")
        skipped += 1
        continue
    faces = extract_faces(path)
    if not faces:
        print(f"  SKIP (no faces): {os.path.basename(path)}")
        skipped += 1
        continue
    y = 1.0 if label == "FAKE" else 0.0
    for f in faces:
        samples.append((f, y))
    print(f"  OK [{label}] {os.path.basename(path)} — {len(faces)} faces")

print(f"\nTotal face samples: {len(samples)}  |  Skipped videos: {skipped}")

if len(samples) < 4:
    print("Not enough samples to fine-tune. Need at least 4 face crops.")
    sys.exit(1)

# ── Dataset ────────────────────────────────────────────────
class FaceDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        x, y = self.samples[i]
        return x, torch.tensor([y], dtype=torch.float32)

loader = DataLoader(FaceDataset(samples), batch_size=BATCH_SIZE, shuffle=True)

# ── Load model ─────────────────────────────────────────────
class MyResNeXt(models.resnet.ResNet):
    def __init__(self):
        super().__init__(block=models.resnet.Bottleneck,
                         layers=[3, 4, 6, 3], groups=32, width_per_group=4)
        self.fc = nn.Linear(2048, 1)

model = MyResNeXt().to(device)
model.load_state_dict(torch.load(os.path.join(BASE_DIR, "resnext.pth"), map_location=device))

# Freeze all layers except layer4 + fc (fine-tune only top layers)
for name, param in model.named_parameters():
    param.requires_grad = name.startswith("layer4") or name.startswith("fc")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable:,}")

# ── Training ───────────────────────────────────────────────
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR
)
criterion = nn.BCEWithLogitsLoss()

print("\nFine-tuning...\n")
model.train()

for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    correct    = 0
    total      = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds   = (torch.sigmoid(out) >= 0.5).float()
        correct += (preds == y).sum().item()
        total   += x.size(0)

    acc = correct / total * 100
    print(f"Epoch {epoch}/{EPOCHS}  loss={total_loss/total:.4f}  acc={acc:.1f}%")

# ── Save ───────────────────────────────────────────────────
out_path = os.path.join(BASE_DIR, "resnext.pth")
torch.save(model.state_dict(), out_path)
print(f"\nSaved fine-tuned model → {out_path}")
print("Restart api.py to use the updated model.")
