# utils/dataset_loader.py
import os
import numpy as np
import tensorflow as tf
from glob import glob
from utils.config import PROCESSED_DIR, BATCH_SIZE, IMG_SIZE

AUTOTUNE = tf.data.AUTOTUNE


def load_npy(path):
    """Reads .npy file and returns float32 array."""
    if isinstance(path, bytes):
        path = path.decode("utf-8")
    elif isinstance(path, np.ndarray):
        path = path.item().decode("utf-8") if path.dtype.type is np.bytes_ else str(path)
    elif hasattr(path, "numpy"):
        path = path.numpy().decode("utf-8")

    arr = np.load(path).astype(np.float32)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, -1)
    return arr


def parse_pair(image_path, mask_path):
    """Reads an image-mask pair."""
    image = tf.numpy_function(load_npy, [image_path], tf.float32)
    mask = tf.numpy_function(load_npy, [mask_path], tf.float32)

    image.set_shape((IMG_SIZE[0], IMG_SIZE[1], None))
    mask.set_shape((IMG_SIZE[0], IMG_SIZE[1], None))

    # Normalize image (0-1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    # Ensure mask is binary
    mask = tf.cast(mask > 0.5, tf.float32)

    return image, mask


def dataset_from_dirs(split="train", batch=BATCH_SIZE, shuffle=True):
    """Creates TensorFlow dataset from processed .npy tiles."""
    image_dir = os.path.join(PROCESSED_DIR, split, "images")
    mask_dir = os.path.join(PROCESSED_DIR, split, "masks")

    images = sorted(glob(os.path.join(image_dir, "*.npy")))
    masks = sorted(glob(os.path.join(mask_dir, "*.npy")))

    assert len(images) == len(masks), f"[{split}] Image-mask count mismatch: {len(images)} vs {len(masks)}"

    img_ds = tf.data.Dataset.from_tensor_slices(images)
    mask_ds = tf.data.Dataset.from_tensor_slices(masks)
    ds = tf.data.Dataset.zip((img_ds, mask_ds))

    if shuffle and split == "train":
        ds = ds.shuffle(len(images))

    ds = ds.map(parse_pair, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch).prefetch(AUTOTUNE)

    return ds


def load_datasets(batch=BATCH_SIZE):
    """
    Returns train, val, and test datasets.
    Example:
        train_ds, val_ds, test_ds = load_datasets()
    """
    print("🔹 Loading train dataset...")
    train_ds = dataset_from_dirs("train", batch=batch, shuffle=True)

    print("🔹 Loading validation dataset...")
    val_ds = dataset_from_dirs("val", batch=batch, shuffle=False)

    print("🔹 Loading test dataset...")
    test_ds = dataset_from_dirs("test", batch=batch, shuffle=False)

    return train_ds, val_ds, test_ds


# ─────────────────────────────
# Quick test
# ─────────────────────────────
if __name__ == "__main__":
    train_ds, val_ds, test_ds = load_datasets()

    for images, masks in train_ds.take(1):
        print("✅ Train batch — Images:", images.shape, "Masks:", masks.shape)
    for images, masks in val_ds.take(1):
        print("✅ Val batch — Images:", images.shape, "Masks:", masks.shape)
    for images, masks in test_ds.take(1):
        print("✅ Test batch — Images:", images.shape, "Masks:", masks.shape)
