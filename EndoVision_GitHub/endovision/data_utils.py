"""Dataset discovery, stratified splitting, and TensorFlow input pipeline."""
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from .config import DATASET_DIR, IMG_SIZE, BATCH_SIZE, SEED, CLASS_NAMES, VALID_EXTS

AUTOTUNE = tf.data.AUTOTUNE


def normalize_name(name):
    return name.lower().replace(" ", "").replace("-", "").replace("_", "")


def find_class_folder(root, class_name):
    wanted = normalize_name(class_name)
    matches = [p for p in Path(root).rglob("*") if p.is_dir()
               and normalize_name(p.name) == wanted]
    if not matches:
        raise FileNotFoundError(
            "Cannot find folder '{}'. Check DATASET_DIR and folder names.".format(class_name))
    return sorted(matches, key=lambda p: len(p.parts))[0]


def collect_images(root=DATASET_DIR):
    paths, labels = [], []
    for label, class_name in enumerate(CLASS_NAMES):
        folder = find_class_folder(root, class_name)
        files = sorted(p for p in folder.rglob("*")
                       if p.is_file() and p.suffix.lower() in VALID_EXTS)
        print("{}: {} images".format(class_name, len(files)))
        paths.extend(str(p) for p in files)
        labels.extend([label] * len(files))
    if not paths:
        raise ValueError("No images found in dataset folders.")
    return np.asarray(paths), np.asarray(labels, dtype=np.int32)


def split_data(paths, labels):
    dev_paths, test_paths, dev_labels, test_labels = train_test_split(
        paths, labels, test_size=0.25, random_state=SEED, stratify=labels)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        dev_paths, dev_labels, test_size=0.20, random_state=SEED, stratify=dev_labels)
    print("Train/validation/test:", len(train_paths), len(val_paths), len(test_paths))
    return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels


def load_image(path, label):
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    return tf.cast(image, tf.float32), tf.cast(label, tf.float32)


def make_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    return ds.map(load_image, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
