# Deepfake CNN-LSTM

Video deepfake detection with a hybrid CNN-LSTM pipeline:
- Input: video
- Output: `real` or `fake` + confidence

## Project structure

- `configs/default.yaml`: all paths and hyperparameters
- `src/data_index.py`: build balanced metadata + train/val split
- `src/preprocess.py`: face crop + fixed-length sequence caching
- `src/model.py`: EfficientNet-B0 + LSTM binary classifier
- `src/train.py`: 2-stage training (freeze, then fine-tune)
- `src/evaluate.py`: metrics (accuracy, precision, recall, F1, ROC-AUC, confusion terms)
- `src/inference.py`: predict a single video
- `src/build_celeb_csv.py`: create CSV for Celeb-DF test set

## Setup

```bash
pip install -r requirements.txt
```

## 1) Build metadata and train/val split (Dataset-A)

Uses:
- `original` as real class
- `DeepFakeDetection`, `Deepfakes`, `Face2Face`, `FaceShifter`, `FaceSwap`, `NeuralTextures` as fake
- samples `167` from each fake folder (1002 fake total)

```bash
python src/data_index.py --config configs/default.yaml --step all
```

## 2) Preprocess train and val into cached tensors

```bash
python src/preprocess.py --config configs/default.yaml --csv "C:/Users/susov/Downloads/archive/deepfake-cnn-lstm/data/train.csv"
python src/preprocess.py --config configs/default.yaml --csv "C:/Users/susov/Downloads/archive/deepfake-cnn-lstm/data/val.csv"
```

This creates:
- `cache/train/index.csv`
- `cache/val/index.csv`

## 3) Train baseline model

```bash
python src/train.py --config configs/default.yaml
```

Best model:
- `checkpoints/best.pt`

Training history:
- `checkpoints/history.json`

## 4) Evaluate on validation set

```bash
python src/evaluate.py --config configs/default.yaml --index_csv "C:/Users/susov/Downloads/archive/deepfake-cnn-lstm/cache/val/index.csv" --checkpoint "C:/Users/susov/Downloads/archive/deepfake-cnn-lstm/checkpoints/best.pt"
```

## 5) External final test on Celeb-DF v2

Keep Celeb-DF fully unseen during training/tuning.

If Celeb-DF folders are:
- `<celeb_root>/real`
- `<celeb_root>/fake`

```bash
python src/build_celeb_csv.py --config configs/default.yaml --celeb_root "D:/datasets/Celeb-DF-v2"
python src/preprocess.py --config configs/default.yaml --csv "C:/Users/susov/Downloads/archive/deepfake-cnn-lstm/data/celeb_test.csv"
python src/evaluate.py --config configs/default.yaml --index_csv "C:/Users/susov/Downloads/archive/deepfake-cnn-lstm/cache/celeb_test/index.csv" --checkpoint "C:/Users/susov/Downloads/archive/deepfake-cnn-lstm/checkpoints/best.pt"
```

## 6) Inference on one video

```bash
python src/inference.py --config configs/default.yaml --checkpoint "C:/Users/susov/Downloads/archive/deepfake-cnn-lstm/checkpoints/best.pt" --video "D:/sample.mp4"
```
