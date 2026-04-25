"""
api.py — Flask backend for deepfake detection
Ensemble: ResNeXt-50 and EfficientNet-B4
Run: python api.py
"""

import os, sys, sqlite3, uuid
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DB_PATH          = os.path.join(BASE_DIR, "feedback.db")
SAVED_VIDEOS_DIR = os.path.join(BASE_DIR, "saved_videos")
os.makedirs(SAVED_VIDEOS_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

_model_loaded = False

def load_model():
    global resnext, efficientnet, get_faces_tensor, torch, device, _model_loaded
    if _model_loaded:
        return

    import torch as _torch
    import torch.nn as nn
    import torchvision.models as models
    import cv2, numpy as np
    from torchvision.transforms import Normalize
    from PIL import Image
    from torchvision import transforms as T

    torch  = _torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    DFDC_DIR = os.path.join(BASE_DIR, "FaceDetector_PP", "FaceDetector_PP")
    sys.path.insert(0, DFDC_DIR)
    sys.path.insert(0, os.path.join(DFDC_DIR, "external", "Pytorch_Retinaface"))
    from face_utils import norm_crop, FaceDetector

    retinaface = FaceDetector(device=device.type)
    retinaface.load_checkpoint(os.path.join(DFDC_DIR, "pth_fiels", "FaceForensics_PP.pth"))

    mean      = [0.485, 0.456, 0.406]
    std       = [0.229, 0.224, 0.225]
    normalize = Normalize(mean, std)
    to_tensor = T.ToTensor()

    # ── ResNeXt-50 ────────────────────────────────────────
    class MyResNeXt(models.resnet.ResNet):
        def __init__(self):
            super().__init__(block=models.resnet.Bottleneck,
                             layers=[3, 4, 6, 3], groups=32, width_per_group=4)
            self.fc = nn.Linear(2048, 1)

    _resnext = MyResNeXt().to(device)
    _resnext.load_state_dict(torch.load(os.path.join(BASE_DIR, "FaceForensics.pth"), map_location=device))
    _resnext.eval()

    # ── EfficientNet-B4 ───────────────────────────────────
    # Try loading a saved weights file; fall back to pretrained ImageNet
    # and add a binary head if no checkpoint exists.
    eff_path = os.path.join(BASE_DIR, "efficientnet_b4.pth")
    try:
        from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
        _eff = efficientnet_b4(weights=None)
        _eff.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(_eff.classifier[1].in_features, 1),
        )
        if os.path.exists(eff_path):
            _eff.load_state_dict(torch.load(eff_path, map_location=device))
            print("EfficientNet-B4 weights loaded from file.")
        else:
            # No custom weights — load ImageNet pretrained backbone, random head
            _eff = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
            _eff.classifier = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(_eff.classifier[1].in_features, 1),
            )
            print("EfficientNet-B4: no checkpoint found, using pretrained backbone (lower accuracy).")
        _eff = _eff.to(device)
        _eff.eval()
        _eff_available = True
    except Exception as e:
        print(f"EfficientNet load failed: {e}. Using ResNeXt only.")
        _eff = None
        _eff_available = False

    # EfficientNet uses slightly different input size (380x380 for B4)
    normalize_eff = Normalize(mean, std)
    resize_eff    = T.Resize((380, 380))

    def _get_faces_tensor(video_path, input_size=224, face_limit=64, frame_skip=9):
        cap   = cv2.VideoCapture(video_path)
        faces_resnext = []
        faces_eff     = []
        while len(faces_resnext) < face_limit:
            for _ in range(frame_skip):
                cap.grab()
            success, img = cap.read()
            if not success:
                break
            boxes, landms = retinaface.detect(img)
            if boxes.shape[0] == 0:
                continue
            areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            order = areas.argmax()
            landmarks = landms[order].numpy().reshape(5, 2).astype(int)
            cropped   = norm_crop(img, landmarks, image_size=input_size)
            aligned   = Image.fromarray(cropped[:, :, ::-1])

            # ResNeXt tensor (224x224)
            faces_resnext.append(normalize(to_tensor(aligned)))

            # EfficientNet tensor (380x380)
            if _eff_available:
                faces_eff.append(normalize_eff(to_tensor(resize_eff(aligned))))

        cap.release()
        if not faces_resnext:
            return None, None, 0

        t_resnext = torch.stack(faces_resnext).to(device)
        t_eff     = torch.stack(faces_eff).to(device) if faces_eff else None
        return t_resnext, t_eff, len(faces_resnext)

    resnext          = _resnext
    efficientnet     = _eff
    get_faces_tensor = _get_faces_tensor
    _model_loaded    = True
    print("Models loaded.")


# ── DB helpers ────────────────────────────────────────────
def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT,
            video_path  TEXT UNIQUE,
            prediction  TEXT,
            true_label  TEXT,
            confidence  REAL,
            p_resnext   REAL,
            feedback    TEXT
        )
    """)
    con.commit()
    con.close()


# ── Routes ────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    load_model()

    # Accept video file upload or local file path
    video_path = None
    tmp_file   = None

    if "video" in request.files:
        f         = request.files["video"]
        unique_name = f"{uuid.uuid4().hex}.mp4"
        saved_path  = os.path.join(SAVED_VIDEOS_DIR, unique_name)
        f.save(saved_path)
        video_path = saved_path
    elif request.json and "path" in request.json:
        video_path = request.json["path"]
    else:
        return jsonify({"error": "No video provided"}), 400

    with torch.no_grad():
        t_resnext, t_eff, n = get_faces_tensor(video_path)
        if t_resnext is None:
            return jsonify({"error": "No faces detected"}), 422

        # ResNeXt score
        p_resnext = torch.sigmoid(resnext(t_resnext).squeeze())
        score_resnext = p_resnext[:n].mean().item() if p_resnext.dim() > 0 else p_resnext.item()

        # EfficientNet score (if available)
        if efficientnet is not None and t_eff is not None:
            p_eff = torch.sigmoid(efficientnet(t_eff).squeeze())
            score_eff = p_eff[:n].mean().item() if p_eff.dim() > 0 else p_eff.item()
            # Weighted ensemble: ResNeXt 60%, EfficientNet 40%
            score = 0.6 * score_resnext + 0.4 * score_eff
        else:
            score     = score_resnext
            score_eff = None

        label = "FAKE" if score >= 0.5 else "REAL"

    return jsonify({
        "result":       label,
        "confidence":   round(score, 4),
        "p_resnext":    round(score_resnext, 4),
        "p_efficientnet": round(score_eff, 4) if score_eff is not None else None,
        "video_path":   video_path,
    })


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json
    required = ["video_path", "prediction", "true_label", "confidence", "feedback"]
    if not all(k in data for k in required):
        return jsonify({"error": "Missing fields"}), 400

    CONF_THRESHOLD = 0.85
    confidence     = data["confidence"]
    fb             = data["feedback"]   # "correct" or "wrong"
    clearly_fake   = confidence > CONF_THRESHOLD
    clearly_real   = confidence < (1 - CONF_THRESHOLD)

    should_store = fb == "wrong" or (fb == "correct" and (clearly_fake or clearly_real))

    if should_store:
        # Move video to sorted folder: saved_videos/{feedback}/{true_label}/
        old_path   = data["video_path"]
        true_label = data["true_label"]
        dest_dir   = os.path.join(SAVED_VIDEOS_DIR, fb, true_label)
        os.makedirs(dest_dir, exist_ok=True)

        new_path = old_path  # default: keep original path
        if os.path.exists(old_path):
            filename = os.path.basename(old_path)
            new_path = os.path.join(dest_dir, filename)
            if os.path.abspath(old_path) != os.path.abspath(new_path):
                import shutil
                shutil.move(old_path, new_path)

        con = sqlite3.connect(DB_PATH)
        existing = con.execute(
            "SELECT id FROM feedback WHERE video_path=?", (old_path,)
        ).fetchone()
        if existing:
            con.execute("""
                UPDATE feedback
                SET timestamp=?, prediction=?, true_label=?, confidence=?, p_resnext=?, feedback=?, video_path=?
                WHERE video_path=?
            """, (datetime.now().isoformat(), data["prediction"], true_label,
                  confidence, data.get("p_resnext", confidence), fb, new_path, old_path))
        else:
            con.execute("""
                INSERT INTO feedback
                    (timestamp, video_path, prediction, true_label, confidence, p_resnext, feedback)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (datetime.now().isoformat(), new_path, data["prediction"],
                  true_label, confidence, data.get("p_resnext", confidence), fb))
        con.commit()
        con.close()
        return jsonify({"stored": True})

    return jsonify({"stored": False, "reason": "correct but ambiguous confidence"})


@app.route("/stats", methods=["GET"])
def stats():
    con     = sqlite3.connect(DB_PATH)
    total   = con.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    wrong   = con.execute("SELECT COUNT(*) FROM feedback WHERE feedback='wrong'").fetchone()[0]
    correct = con.execute("SELECT COUNT(*) FROM feedback WHERE feedback='correct'").fetchone()[0]
    con.close()
    return jsonify({"total": total, "wrong": wrong, "correct": correct,
                    "ready_to_retrain": total >= 50})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    init_db()
    print("Starting DeepFake Detection API on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
