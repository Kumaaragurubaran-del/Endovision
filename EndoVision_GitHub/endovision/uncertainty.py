"""Monte Carlo Dropout prediction and validation-only threshold selection."""
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score


def mc_predict(model, dataset, passes=30):
    feature_model = tf.keras.Model(model.input, model.get_layer("gap").output)
    dropout_layer = model.get_layer("head_dropout")
    prediction_layer = model.get_layer("prediction")
    means, uncertainties = [], []
    for images, _ in dataset:
        pooled = feature_model(images, training=False)
        pass_predictions = []
        for _ in range(passes):
            dropped = dropout_layer(pooled, training=True)
            probabilities = prediction_layer(dropped, training=False)
            pass_predictions.append(tf.squeeze(probabilities, axis=-1).numpy())
        pass_predictions = np.stack(pass_predictions, axis=0)
        means.extend(np.mean(pass_predictions, axis=0).tolist())
        uncertainties.extend(np.std(pass_predictions, axis=0).tolist())
    return np.asarray(means), np.asarray(uncertainties)


def select_threshold(probabilities, labels):
    thresholds = np.arange(0.30, 0.701, 0.01)
    scores = [f1_score(labels, probabilities >= t, zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])
