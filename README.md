# TRUEVISION — Real-Time Deepfake Detection Overlay

A desktop overlay app that detects deepfake videos in real-time using an ensemble of deep learning models. Capture any screen content, analyze it , and provide feedback to continuously improve the model.

---

## Demo

> Capture a video playing on screen → AI analyzes it → Get REAL ✅ or FAKE ❌ result with confidence score → Give feedback → Model improves over time.

---

## Features

- **Always-on-top overlay** — works on top of any app (Instagram, YouTube, etc.)
- **Screen capture** — record any window or display source
- **File upload** — analyze local video files directly
- **Ensemble model** — ResNeXt-50 + EfficientNet-B4 for higher accuracy
- **Face detection** — RetinaFace for precise face localization
- **Feedback loop** — correct/wrong feedback auto-sorts videos for fine-tuning
- **Fine-tuning script** — retrain the model on your own labeled data

---

## Architecture

```
┌─────────────────────────┐         ┌──────────────────────────────┐
│   Electron Frontend     │  HTTP   │      Flask Backend           │
│                         │ ──────► │                              │
│  • Screen capture       │         │  • RetinaFace (face detect)  │
│  • Always-on-top UI     │ ◄────── │  • ResNeXt-50                │
│  • Result display       │         │  • EfficientNet-B4           │
│  • Feedback buttons     │         │  • Ensemble scoring          │
└─────────────────────────┘         │  • SQLite feedback DB        │
                                    │  • Video sorting             │
                                    └──────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Electron, HTML/CSS/JS |
| Backend | Python, Flask |
| Models | PyTorch, ResNeXt-50, EfficientNet-B4 |
| Face Detection | RetinaFace |
| Database | SQLite |

---

## Project Structure

```
├── backend/
│   ├── api.py                  # Flask API (predict, feedback, stats)
│   ├── finetune.py             # Fine-tune model on feedback data
│   ├── sort_feedback_videos.py # Sort videos by feedback label
│   ├── blazeface.py            # Blazeface face detector
│   ├── deepfake-cnn-lstm/      # CNN-LSTM model source
│   └── kaggle-dfdc-master/     # DFDC model & RetinaFace
│
├── frontend/
│   ├── deepfake_detector_ui.html  # Main UI
│   ├── main.js                    # Electron main process
│   ├── preload.js                 # Electron preload bridge
│   └── package.json
│
└── README.md
```

---

## Setup

### Backend

```bash
cd backend
pip install flask flask-cors torch torchvision opencv-python pillow
python api.py
```

> Requires `resnext.pth` model weights in `backend/`. Not included in repo due to file size.

### Frontend

```bash
cd frontend
npm install
npm start
```

---

## Model Weights

The following files are required but not included in this repo (too large for GitHub):

| File | Description |
|------|-------------|
| `backend/resnext.pth` | ResNeXt-50 deepfake detector |
| `backend/kaggle-dfdc-master/.../RetinaFace-Resnet50-fixed.pth` | RetinaFace weights |
| `backend/efficientnet_b4.pth` | EfficientNet-B4 (optional, for ensemble) |

---

## How the Feedback Loop Works

```
User gives feedback (Correct / Wrong)
           │
           ▼
    feedback == "wrong"  ──────────────────────► Always stored
           │
    feedback == "correct"
           │
    confidence > 0.85 or < 0.15  ──────────────► Stored (high confidence)
           │
    0.15 < confidence < 0.85  ──────────────────► Skipped (ambiguous)
```

Videos are automatically sorted into:
```
backend/saved_videos/
  correct/FAKE/   ← confirmed fakes (high confidence)
  correct/REAL/   ← confirmed reals (high confidence)
  wrong/FAKE/     ← model said REAL, actually FAKE
  wrong/REAL/     ← model said FAKE, actually REAL
```

### Fine-tuning

```bash
cd backend
python finetune.py
```

Trains only the last layers (`layer4` + `fc`) of ResNeXt on your labeled feedback data.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Upload video, get FAKE/REAL result |
| POST | `/feedback` | Submit correct/wrong feedback |
| GET | `/stats` | Get feedback statistics |
| GET | `/health` | Backend health check |

---


---


