# AED: Within-Group Ranking of Art-Exam Works

AED contains 3,360 artworks from 93 examination groups across Sketch, Quick Sketch, and Color Painting. This repository provides the dataset, MLVS-Net, and evaluation code for within-group quality ranking.

## Contents

- `data/`: image archive and score labels.
- `splits/`: fixed training, validation, and test groups.
- `configs/`: experiment settings.
- `train.py`: model training.
- `evaluate.py`: performance evaluation.
- `THIRD_PARTY.md`, `licenses/`: sources and licence notices.

## Environment

- Python 3.8
- PyTorch 2.1.0
- torchvision 0.16.0
- CUDA 11.8

```bash
python -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```
