"""Project settings. Update DATASET_DIR to point to your dataset folder."""
from pathlib import Path

DATASET_DIR = Path("dataset")
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
HEAD_EPOCHS = 10
FINE_TUNE_EPOCHS = 15
MC_PASSES = 30
SEED = 42
CLASS_NAMES = ["no_endometriosis", "endometriosis"]  # label 0, then label 1
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODEL_PATH = Path("artifacts/best_endometriosis.keras")
RESULTS_PATH = Path("outputs/endovision_uncertainty_results.csv")
