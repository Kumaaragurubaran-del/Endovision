# EndoVision: Uncertainty-Aware Endometriosis Image Classification

EndoVision is a research prototype that classifies medical images into `endometriosis` and `no_endometriosis` categories using EfficientNetB0 transfer learning. It also estimates prediction variability with Monte Carlo Dropout and provides Grad-CAM visualizations to inspect image regions influencing a prediction.

> **Important:** This project is for educational/research use only. It is not a clinically validated diagnostic tool and must not be used to make medical decisions.

## Features

- EfficientNetB0 pretrained on ImageNet
- Image augmentation and two-stage transfer learning
- Stratified train/validation/test split (60% / 15% / 25%)
- Monte Carlo Dropout uncertainty estimate (30 stochastic passes by default)
- Validation-based classification threshold and uncertainty review cutoff
- Accuracy, balanced accuracy, F1-score, ROC-AUC, confusion matrix and classification report
- CSV export of test predictions and review flags
- Grad-CAM visualization for qualitative interpretation

## Repository structure

```text
EndoVision_GitHub/
├── endovision/
│   ├── __init__.py
│   ├── config.py          # paths and training settings
│   ├── data_utils.py      # image discovery, split and input pipeline
│   ├── model.py           # EfficientNetB0 model and callbacks
│   ├── uncertainty.py     # Monte Carlo Dropout and threshold selection
│   └── explainability.py  # Grad-CAM visualization
├── dataset/               # your local dataset (not committed to GitHub)
├── artifacts/             # saved model (not committed to GitHub)
├── outputs/               # evaluation CSV (not committed to GitHub)
├── train.py               # main training/evaluation script
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset setup

Create this folder structure locally. Folder names are case-insensitive and spaces, hyphens, and underscores are normalized.

```text
dataset/
├── endometriosis/
│   ├── image_001.jpg
│   └── ...
└── no_endometriosis/
    ├── image_001.jpg
    └── ...
```

Place the dataset in `dataset/`, or change `DATASET_DIR` in `endovision/config.py`. Do not upload the dataset to GitHub unless you have permission and the license allows redistribution. Large medical datasets are best kept outside the repository.

## Setup

Python 3.10 or 3.11 is recommended for compatibility with TensorFlow versions. Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

## Run

From the repository root:

```bash
python train.py
```

The first run downloads pretrained EfficientNetB0 weights and trains the model. Training can take time and is faster with a compatible GPU. Outputs are written to `artifacts/` and `outputs/`.

## Data split and evaluation

The code reserves 25% of the images for the held-out test set. The remaining 75% is split into 60% training and 15% validation overall. The validation set is used for early stopping, threshold selection, and setting the uncertainty review cutoff. The test set is used for final reporting.

If images are multiple frames/images from the same patient or surgery, use patient-/surgery-level splitting to reduce data leakage. The current implementation performs stratified image-level splitting because patient identifiers are not included in the assumed dataset structure.

## How uncertainty is used

The classifier runs 30 stochastic dropout passes per image. The mean probability is used as the prediction score, while the standard deviation estimates prediction variability. Images exceeding the validation-derived 90th-percentile uncertainty cutoff are marked `manual_review` in the output CSV. This cutoff is a project design choice, not a clinical threshold; uncertainty is not a guarantee that a prediction is correct.

## Output files

- `artifacts/best_endometriosis.keras` — best model checkpoint by validation loss
- `outputs/endovision_uncertainty_results.csv` — true label, predicted label, mean score, uncertainty and review decision

## License and dataset

Add a license appropriate to your own code before public release. Check the source dataset's license and privacy conditions separately; this repository does not include dataset images.
