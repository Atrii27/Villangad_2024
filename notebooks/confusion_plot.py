# ===============================================================
# 🛰️ Confusion Matrix Visualization for 5 U-Net Variants
# ===============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf

# ===============================================================
# CONFIGURATION (ABSOLUTE PATHS FOR WINDOWS)
# ===============================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR,"raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed","test")
MODEL_DIR = os.path.join(ROOT, "experiments_all")
OUT_DIR = MODEL_DIR  # save confusion matrices here

IMG_PATH = os.path.join(PROCESSED_DIR, "images")
MASK_PATH = os.path.join(PROCESSED_DIR, "masks")

# ===============================================================
# LOAD TEST DATA (.NPY FILES)
# ===============================================================
def load_npy_images(img_dir, mask_dir):
    X, Y = [], []
    mask_files = sorted(os.listdir(mask_dir))
    for f in sorted(os.listdir(img_dir)):
        if f.endswith(".npy") and f in mask_files:
            img = np.load(os.path.join(img_dir, f))
            mask = np.load(os.path.join(mask_dir, f))
            X.append(img)
            Y.append(mask)
    return np.array(X), np.array(Y)

print("📂 Loading test data...")
X_test, y_test = load_npy_images(IMG_PATH, MASK_PATH)
print(f"✅ Loaded test set: X={X_test.shape}, y={y_test.shape}")

# Flatten ground truth for confusion matrix
y_true = y_test.reshape(-1)

# ===============================================================
# LOAD MODELS
# ===============================================================
models_dict = {
    "U-Net": os.path.join(MODEL_DIR, "unet.keras"),
    "ResU-Net": os.path.join(MODEL_DIR, "resunet.keras"),
    "Attn U-Net": os.path.join(MODEL_DIR, "attnunet.keras"),
    "Attn ResU-Net": os.path.join(MODEL_DIR, "attnresunet.keras"),
    "ASDMS U-Net": os.path.join(MODEL_DIR, "asdms.keras")
}

loaded_models = {}
for name, path in models_dict.items():
    if os.path.exists(path):
        print(f"🔄 Loading model: {name}")
        loaded_models[name] = tf.keras.models.load_model(path, compile=False)
    else:
        print(f"⚠️ Model file not found: {path}")

# ===============================================================
# FUNCTION TO PLOT CONFUSION MATRIX
# ===============================================================
def plot_conf_mat(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=["Non-Landslide", "Landslide"],
                yticklabels=["Non-Landslide", "Landslide"])
    plt.title(f"Confusion Matrix — {model_name}", fontsize=12, weight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    save_path = os.path.join(OUT_DIR, f"confmat_{model_name.replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved: {save_path}")

# ===============================================================
# GENERATE CONFUSION MATRICES
# ===============================================================
for name, model in loaded_models.items():
    print(f"\n🔍 Generating confusion matrix for {name} ...")
    y_pred = model.predict(X_test, verbose=0)
    y_pred_bin = (y_pred > 0.5).astype(np.uint8).reshape(-1)
    plot_conf_mat(y_true, y_pred_bin, name)

print("\n✅ All confusion matrices saved successfully.")
