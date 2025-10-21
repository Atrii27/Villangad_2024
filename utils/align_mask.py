import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling as Resample
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

image_path = os.path.join(RAW_DIR, "20241110_053942_45_24f7_3B_AnalyticMS_SR_8b_clip.tif")
mask_path  = os.path.join(RAW_DIR, "mask.tif")
out_path   = os.path.join(RAW_DIR, "mask_aligned.tif")

with rasterio.open(image_path) as ref, rasterio.open(mask_path) as src:
    profile = ref.profile
    profile.update(count=1, dtype=rasterio.uint8)

    mask_data = src.read(1)
    aligned = np.zeros((ref.height, ref.width), dtype=np.uint8)

    reproject(
        source=mask_data,
        destination=aligned,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=ref.transform,
        dst_crs=ref.crs,
        resampling=Resample.nearest
    )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(aligned, 1)

print("✅ Saved aligned mask to:", out_path)
