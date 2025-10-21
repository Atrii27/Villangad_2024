# utils/config.py
import os

# === PATH CONFIGURATION ===
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR,"raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")

# === MODEL / TRAINING PARAMETERS ===
IMG_SIZE = (256, 256)        # Tile size
BATCH_SIZE = 16              # Number of samples per batch
N_BANDS = 8                  # PlanetScope has 8 bands in AnalyticMS_SR_8b product
NUM_CLASSES = 1              # Binary segmentation
EPOCHS = 150
LEARNING_RATE = 5e-4         # Pass this value to Adam(learning_rate=LEARNING_RATE)

# === OUTPUT DIRECTORIES ===
CHECKPOINT_DIR = os.path.join(OUTPUTS_DIR, "checkpoints", "unet")
LOG_DIR = os.path.join(OUTPUTS_DIR, "logs", "unet")
PRED_DIR = os.path.join(OUTPUTS_DIR, "predictions", "unet")

# === CREATE FOLDERS IF NOT EXIST ===
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
