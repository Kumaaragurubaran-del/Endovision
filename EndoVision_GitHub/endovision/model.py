"""EfficientNetB0 transfer-learning model and training callbacks."""
import tensorflow as tf
from .config import IMG_SIZE, MODEL_PATH


def build_model():
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.04),
        tf.keras.layers.RandomZoom(0.08),
        tf.keras.layers.RandomContrast(0.08),
    ], name="augmentation")

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
    base_model.trainable = False
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3), name="image")
    x = augmentation(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(0.35, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="prediction")(x)
    model = tf.keras.Model(inputs, outputs)
    compile_model(model, 3e-4)
    return model


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])


def callbacks(patience, min_lr):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(str(MODEL_PATH), monitor="val_loss",
                                           save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                             patience=2, min_lr=min_lr, verbose=1)
    ]
