# AED: Within-Group Ranking of Art-Exam Works

AED contains 3,360 artworks from 93 examination groups across Sketch, Quick Sketch, and Color Painting. This repository provides the dataset, MLVS-Net, and evaluation code for within-group quality ranking.

## Contents

- `data/`: image archive and score labels.
- `splits/`: fixed training, validation, and test groups.
- `configs/`: experiment settings.
- `train.py`: model training.
- `evaluate.py`: performance evaluation.
- `THIRD_PARTY.md`, `licenses/`: sources and licence notices.

## Layer-configuration validation

The layer combinations `gap1`, `gap2`, `gap3`, and `gap4` are configuration candidates. They are restricted to internal-validation training and reporting. The command below writes per-epoch validation histories and does not evaluate or write outer-test predictions:

```bash
python train.py --models gap1 gap2 gap3 gap4 --seeds 0 1 2 --folds 0 1 2 3 4 \
  --image-root /path/to/images --cache /path/to/cache --assets /path/to/assets \
  --output outputs/layer_validation
```

`train.py` rejects `--evaluate-test` whenever any `gap1`--`gap4` candidate is requested. In this mode the program constructs only the TRAIN and internal-VAL subsets; the outer-TEST rows are not attached to the training jobs.

The revised validation table can be reproduced from the 60 fixed history files without reading prediction or metric files from TEST:

```bash
python summarize_validation.py --run outputs/layer_validation \
  --output outputs/layer_validation_summary
```

The summarizer reports the epoch-80 internal-validation metrics for all four candidates. It does not select a configuration from outer-TEST performance.

## Final outer-test evaluation

After the feature-layer configuration is fixed, the selected architecture is addressed only by the final model name `mlvs`. Outer-test evaluation requires the explicit flag below:

```bash
python train.py --models mlvs --seeds 0 1 2 --folds 0 1 2 3 4 \
  --image-root /path/to/images --cache /path/to/cache --assets /path/to/assets \
  --evaluate-test --output outputs/mlvs_final_test
```

Without `--evaluate-test`, all neural models run in internal-validation-only mode.

`evaluate.py` also rejects files whose model field is `gap1`, `gap2`, `gap3`, or `gap4`. Thus candidate layer combinations cannot be inserted into the final outer-TEST summary through the standalone evaluator.

## Environment

- Python 3.8
- PyTorch 2.1.0
- torchvision 0.16.0
- CUDA 11.8

```bash
python -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```
