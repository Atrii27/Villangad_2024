## utils/preprocessing.py
import os
import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.model_selection import train_test_split
from utils.config import RAW_DIR, PROCESSED_DIR, IMG_SIZE

# Input file names
IMAGE_TIF = os.path.join(RAW_DIR, "20241110_053942_45_24f7_3B_AnalyticMS_SR_8b_clip.tif")
MASK_TIF  = os.path.join(RAW_DIR, "mask.tif")

def create_dirs():
    """Creates train/val/test directory structure."""
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(PROCESSED_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(PROCESSED_DIR, split, "masks"), exist_ok=True)

def tile_raster_pair(image_path, mask_path, tile_size=IMG_SIZE, stride=None):
    """Tiles image and mask together so they stay aligned."""
    if stride is None:
        stride = tile_size[0]

    imgs, msks = [], []
    with rasterio.open(image_path) as img_src, rasterio.open(mask_path) as mask_src:
        assert img_src.height == mask_src.height and img_src.width == mask_src.width, "Image and mask must align"
        for top in range(0, img_src.height - tile_size[0] + 1, stride):
            for left in range(0, img_src.width - tile_size[1] + 1, stride):
                window = Window(left, top, tile_size[1], tile_size[0])
                img = np.moveaxis(img_src.read(window=window), 0, 2).astype(np.float32)
                mask = mask_src.read(1, window=window).astype(np.uint8)  # single band mask
                # Normalize image to 0-1
                img = img / (np.max(img) + 1e-8)
                imgs.append(img)
                msks.append(np.expand_dims(mask, -1))
    return imgs, msks

def save_split(images, masks):
    """Splits into train/val/test and saves npy tiles."""
    X_train, X_tmp, y_train, y_tmp = train_test_split(images, masks, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

    splits = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }

    for split, (X, y) in splits.items():
        img_dir = os.path.join(PROCESSED_DIR, split, "images")
        mask_dir = os.path.join(PROCESSED_DIR, split, "masks")
        for i, (img, msk) in enumerate(zip(X, y)):
            np.save(os.path.join(img_dir, f"{split}_{i:05d}.npy"), img)
            np.save(os.path.join(mask_dir, f"{split}_{i:05d}.npy"), msk)
        print(f"Saved {len(X)} tiles in {split}/images and {split}/masks")

def main():
    create_dirs()
    print("Tiling and splitting image + mask...")
    imgs, msks = tile_raster_pair(IMAGE_TIF, MASK_TIF)
    print(f"Total tiles generated: {len(imgs)}")
    save_split(imgs, msks)
    print("✅ Done! Data ready under processed/train, val, test folders.")

if __name__ == "__main__":
    main()
