# TrueVision — Real-Time Deepfake Detection

> A desktop overlay application that detects deepfake videos in real-time using an ensemble of deep learning models, with a built-in feedback loop for continuous model improvement.

---

## Overview

TrueVision sits on top of any application as a floating overlay. Point it at any video — whether streaming on YouTube, Instagram, or a local file — and it will tell you if the video is real or fake, along with a confidence score.

---

## Features

| Feature | Description |
|---------|-------------|
| Always-on-top overlay | Works over any app without interrupting your workflow |
| Screen capture | Record and analyze any window or display source |
| File upload | Analyze local video files directly |
| Ensemble detection | Multiple models combined for higher accuracy |
| Face detection | Precise face localization before classification |
| Feedback loop | User corrections are stored and used to improve the model |
| Retraining | Fine-tune the model on collected feedback data |

---

## Architecture

```
┌─────────────────────────┐         ┌──────────────────────────────┐
│   Electron Frontend     │  HTTP   │      Flask Backend           │
│                         │ ──────► │                              │
│  • Screen capture       │         │  • Face detection            │
│  • Always-on-top UI     │ ◄────── │  • Deepfake classification   │
│  • Result display       │         │  • Confidence scoring        │
│  • Feedback buttons     │         │  • SQLite feedback DB        │
└─────────────────────────┘         │  • Video sorting             │
                                    └──────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Electron, HTML/CSS/JS |
| Backend | Python, Flask |
| Models | PyTorch |
| Face Detection | RetinaFace |
| Database | SQLite |

---

## Project Structure

```
TrueVision/
│
├── backend/
│   ├── api.py                     # Flask API — predict, feedback, stats
│   ├── retrain.py                 # Fine-tune model on feedback data
│   ├── sort_feedback_videos.py    # Utility to sort videos by feedback label
│   ├── FaceForensics.pth          # Primary model weights (not in repo)
│   └── FaceDetector_PP/           # RetinaFace face detector
│       └── FaceDetector_PP/
│           ├── face_utils.py
│           ├── pth_fiels/
│           │   └── FaceForensics_PP.pth   # Face detector weights (not in repo)
│           └── external/
│               └── Pytorch_Retinaface/    # RetinaFace source code
│
├── frontend/
│   ├── deepfake_detector_ui.html  # Main application UI
│   ├── fab.html                   # Floating action button overlay
│   ├── main.js                    # Electron main process
│   ├── preload.js                 # Electron preload bridge
│   └── package.json
│
├── eval_results/                  # Model evaluation charts & metrics
├── start.bat                      # Quick start script (Windows)
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.8+
- Node.js 16+

### Backend

```bash
cd backend
pip install flask flask-cors torch torchvision opencv-python pillow scikit-image
python api.py
```

### Frontend

```bash
cd frontend
npm install
npm start
```

> **Note:** Model weight files are not included in this repository due to file size. See [Model Weights](#model-weights) below.

---

## Model Weights

These files must be placed manually before running the app:

| File | Description |
|------|-------------|
| `backend/FaceForensics.pth` | Primary deepfake detection model |
| `backend/FaceDetector_PP/FaceDetector_PP/pth_fiels/FaceForensics_PP.pth` | Face detector weights |
| `backend/efficientnet_b4.pth` | Secondary model — optional, enables ensemble mode |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Upload a video and get FAKE / REAL result |
| `POST` | `/feedback` | Submit correct / wrong feedback |
| `GET` | `/stats` | View feedback statistics |
| `GET` | `/health` | Backend health check |

---

## Feedback & Retraining

### How feedback is stored

```
User clicks "Correct" or "Wrong"
              │
              ▼
       feedback = "wrong" ──────────────────────► Always stored
              │
       feedback = "correct"
              │
       confidence > 0.85 or < 0.15 ─────────────► Stored (clearly fake/real)
              │
       0.15 ≤ confidence ≤ 0.85 ────────────────► Skipped (ambiguous)
              │
              ▼
       50+ samples collected ──────────────────► Ready to retrain
```

### Sorted video structure

```
backend/saved_videos/
  ├── correct/
  │   ├── FAKE/    ← model correctly identified fakes
  │   └── REAL/    ← model correctly identified reals
  └── wrong/
      ├── FAKE/    ← actually FAKE, model said REAL
      └── REAL/    ← actually REAL, model said FAKE
```

### Running retraining

```bash
cd backend
python retrain.py
```

After retraining, rename `resnext_finetuned.pth` → `FaceForensics.pth` and restart `api.py`.

---

## Evaluation Results

Training and evaluation charts are available in `eval_results/`:

- Accuracy & Loss curves
- Confusion matrix
- Classification report
- Score distribution

---

## Author

**Susovan Patra**
- GitHub: [@Susovan12](https://github.com/Susovan12)

---


