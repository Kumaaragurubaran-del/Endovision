"""Train and evaluate EndoVision. Run from the repository root: python train.py"""
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_auc_score, f1_score,
                             balanced_accuracy_score)

from endovision.config import (DATASET_DIR, HEAD_EPOCHS, FINE_TUNE_EPOCHS,
                               MC_PASSES, CLASS_NAMES, MODEL_PATH, RESULTS_PATH, SEED)
from endovision.data_utils import collect_images, split_data, make_dataset
from endovision.model import build_model, compile_model, callbacks
from endovision.uncertainty import mc_predict, select_threshold
from endovision.explainability import show_gradcam


def plot_training(history_head, history_fine):
    for metric in ["accuracy", "loss"]:
        train_values = history_head.history.get(metric, []) + history_fine.history.get(metric, [])
        val_values = history_head.history.get("val_" + metric, []) + history_fine.history.get("val_" + metric, [])
        plt.figure()
        plt.plot(train_values, label="Training")
        plt.plot(val_values, label="Validation")
        plt.xlabel("Epoch"); plt.ylabel(metric)
        plt.title("Training vs validation " + metric)
        plt.legend(); plt.tight_layout(); plt.show()


def main():
    tf.keras.utils.set_random_seed(SEED)
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(
            "Dataset folder not found: {}. Update DATASET_DIR in endovision/config.py.".format(DATASET_DIR))

    paths, labels = collect_images(DATASET_DIR)
    print("Total images:", len(paths))
    train_p, val_p, test_p, train_y, val_y, test_y = split_data(paths, labels)
    train_ds = make_dataset(train_p, train_y, training=True)
    val_ds = make_dataset(val_p, val_y)
    test_ds = make_dataset(test_p, test_y)

    counts = np.bincount(train_y, minlength=2)
    class_weight = {i: len(train_y) / (2.0 * max(counts[i], 1)) for i in range(2)}
    print("Class weights:", class_weight)

    model = build_model()
    print("\nStage 1: Train classification head")
    history_head = model.fit(train_ds, validation_data=val_ds, epochs=HEAD_EPOCHS,
                             class_weight=class_weight, callbacks=callbacks(4, 1e-6))

    print("\nStage 2: Fine-tune upper EfficientNet layers")
    base_model = model.get_layer("efficientnetb0")
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    compile_model(model, 1e-5)
    history_fine = model.fit(train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS,
                             class_weight=class_weight, callbacks=callbacks(5, 1e-7))

    # Reload best validation-loss checkpoint before thresholding/evaluation.
    model = tf.keras.models.load_model(str(MODEL_PATH))
    print("\nEstimating validation uncertainty with Monte Carlo Dropout...")
    val_prob, val_uncertainty = mc_predict(model, val_ds, MC_PASSES)
    threshold = select_threshold(val_prob, val_y)
    uncertainty_cutoff = float(np.percentile(val_uncertainty, 90))
    print("Validation-selected threshold:", round(threshold, 2))
    print("Validation 90th-percentile uncertainty cutoff: {:.5f}".format(uncertainty_cutoff))

    print("\nEstimating held-out test predictions and uncertainty...")
    test_prob, test_uncertainty = mc_predict(model, test_ds, MC_PASSES)
    test_pred = (test_prob >= threshold).astype(int)
    accepted = test_uncertainty <= uncertainty_cutoff
    print("\nHeld-out test results")
    print("Accuracy: {:.4f}".format(np.mean(test_pred == test_y)))
    print("Balanced accuracy: {:.4f}".format(balanced_accuracy_score(test_y, test_pred)))
    print("F1-score: {:.4f}".format(f1_score(test_y, test_pred, zero_division=0)))
    print("ROC-AUC: {:.4f}".format(roc_auc_score(test_y, test_prob)))
    print(classification_report(test_y, test_pred, target_names=CLASS_NAMES,
                                digits=4, zero_division=0))
    print("Automatic-prediction coverage: {:.1f}%".format(100 * np.mean(accepted)))
    if np.any(accepted):
        print("Accuracy on accepted images: {:.4f}".format(np.mean(test_pred[accepted] == test_y[accepted])))
    print("Flagged for manual review: {} / {}".format(np.sum(~accepted), len(test_y)))

    plt.figure()
    ConfusionMatrixDisplay(confusion_matrix(test_y, test_pred), display_labels=CLASS_NAMES).plot(
        values_format="d", xticks_rotation=20)
    plt.title("Endometriosis classification - held-out test set")
    plt.tight_layout(); plt.show()

    plt.figure()
    plt.hist(test_uncertainty[accepted], bins=15, alpha=0.7, label="Automatic prediction")
    if np.any(~accepted):
        plt.hist(test_uncertainty[~accepted], bins=15, alpha=0.7, label="Flagged for review")
    plt.axvline(uncertainty_cutoff, linestyle="--", label="Validation uncertainty cutoff")
    plt.xlabel("Prediction uncertainty (standard deviation)")
    plt.ylabel("Number of images"); plt.title("Uncertainty-aware prediction analysis")
    plt.legend(); plt.tight_layout(); plt.show()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "image_path": test_p,
        "true_label": [CLASS_NAMES[int(x)] for x in test_y],
        "predicted_label": [CLASS_NAMES[int(x)] for x in test_pred],
        "mean_endometriosis_score": test_prob,
        "prediction_uncertainty": test_uncertainty,
        "decision": np.where(accepted, "automatic_prediction", "manual_review")
    }).to_csv(RESULTS_PATH, index=False)
    print("Saved results to:", RESULTS_PATH)

    plot_training(history_head, history_fine)
    if len(test_p):
        show_gradcam(model, test_p[0])
    print("Model saved to:", MODEL_PATH)
    print("Research prototype only; not a clinically validated diagnostic tool.")


if __name__ == "__main__":
    main()
