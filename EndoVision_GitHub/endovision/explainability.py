"""Grad-CAM heatmap for visual inspection (not lesion segmentation)."""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from .config import IMG_SIZE


def show_gradcam(model, image_path):
    base_model = model.get_layer("efficientnetb0")
    conv_layer = next((layer for layer in reversed(base_model.layers)
                       if isinstance(layer, tf.keras.layers.Conv2D)), None)
    if conv_layer is None:
        print("Could not find convolution layer for Grad-CAM.")
        return

    raw = tf.io.read_file(str(image_path))
    image_tensor = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image_tensor = tf.image.resize(image_tensor, IMG_SIZE)
    batch = tf.cast(image_tensor, tf.float32)[None, ...]
    feature_model = tf.keras.Model(
        base_model.input, [base_model.get_layer(conv_layer.name).output, base_model.output])

    with tf.GradientTape() as tape:
        features, base_output = feature_model(batch, training=False)
        pooled = model.get_layer("gap")(base_output)
        dropped = model.get_layer("head_dropout")(pooled, training=False)
        score = model.get_layer("prediction")(dropped)[:, 0]
    gradients = tape.gradient(score, features)
    if gradients is None:
        print("Grad-CAM gradient unavailable for this model graph.")
        return

    weights = tf.reduce_mean(gradients, axis=(1, 2))
    heatmap = tf.reduce_sum(features * weights[:, None, None, :], axis=-1)[0]
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    heatmap = tf.image.resize(heatmap[..., None], IMG_SIZE).numpy().squeeze()
    original = tf.keras.utils.img_to_array(
        tf.keras.utils.load_img(str(image_path), target_size=IMG_SIZE)).astype("uint8")
    probability = float(model.predict(batch, verbose=0)[0][0])

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original); plt.title("Input image"); plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(original); plt.imshow(heatmap, cmap="jet", alpha=0.4)
    plt.title("Grad-CAM | endometriosis score {:.3f}".format(probability)); plt.axis("off")
    plt.tight_layout(); plt.show()
