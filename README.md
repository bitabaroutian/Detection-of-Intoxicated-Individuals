# Drunk / Sober Video Classification

Multimodal pipeline for binary intoxication classification from video: scene segmentation, facial landmarks (EAR/MAR), and a fusion classifier (R3D-18 + GAT-LSTM).

## Pipeline

Run these notebooks **in order**:

| # | Notebook | Output |
|---|----------|--------|
| 1 | [`scene_detection.ipynb`](scene_detection.ipynb) | Face-containing video segments (TransNetV2 + dlib) |
| 2 | [`Landmarks,MAR,EAR_extraction.ipynb`](Landmarks,MAR,EAR_extraction.ipynb) | Landmarks, EAR, MAR CSV (SPIGA) |
| 3 | [`Model.ipynb`](Model.ipynb) | Trained drunk/sober model |

**Fusion model:** R3D-18 on 16-frame clips + GAT-LSTM on 68 landmarks, merged with learnable weighted fusion. Subject-based train/val split to avoid leakage.

## Data layout

Videos and CSVs are not included. Use this structure locally:

```
data/
├── raw_videos/       → step 1 input
├── segments/         → step 1 output / step 2 input
├── output/           → landmarks_ear_mar.csv
└── dataset/
    ├── drunk/
    └── sober/
```

Update paths in each notebook before running. Training cells in notebook 3 are commented until paths are set.

## Setup

- Python 3.10+, CUDA GPU recommended, ffmpeg
- Clone [TransNetV2](https://github.com/soCzech/TransNetV2) and [SPIGA](https://github.com/andresprados/SPIGA) (`pip install -e .` inside SPIGA)
- Install: PyTorch, torch-geometric, OpenCV, pandas, numpy, scikit-learn, dlib, tensorflow

Colab-specific cells (Drive mount, `apt-get`) are commented out; uncomment only if using Colab.
