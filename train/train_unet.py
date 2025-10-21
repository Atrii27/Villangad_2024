# train/train_unet.py
import os, sys, datetime
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau
)

# --- Add project root to sys.path ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# -----------------------------------

from models.unet import build_unet
from utils.dataset_loader import load_datasets
from utils.config import LEARNING_RATE, EPOCHS, CHECKPOINT_DIR, LOG_DIR

# ============================================================
# ⚙️  CONFIGURATION
# ============================================================
EXPERIMENT_NAME = "unet_run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Enable mixed precision (if supported GPU)
try:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print("✅ Mixed precision enabled.")
except Exception as e:
    print("⚠️ Could not enable mixed precision:", e)

# ============================================================
# 🎯  CUSTOM METRICS & LOSS
# ============================================================
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )

def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def dice_bce_loss(y_true, y_pred, smooth=1e-6):
    """Hybrid Dice + BCE loss for better boundary learning."""
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice_loss = 1 - (2. * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return dice_loss + bce

# ============================================================
# 🧠  MAIN TRAINING LOOP
# ============================================================
def main():
    print("\n🚀 Loading datasets...")
    train_ds, val_ds, _ = load_datasets()

    print("\n🏗️  Building U-Net model...")
    model = build_unet()
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=dice_bce_loss,
        metrics=[dice_coef, iou_metric, "accuracy"]
    )

    print("\n🧠 Training started...")
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"unet_best_{EXPERIMENT_NAME}.h5")
    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_iou_metric",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        CSVLogger(os.path.join(LOG_DIR, f"unet_training_log_{EXPERIMENT_NAME}.csv")),
        TensorBoard(log_dir=os.path.join(LOG_DIR, EXPERIMENT_NAME)),
        EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, verbose=1, min_lr=1e-6
        ),
    ]

    # Optionally resume from previous checkpoint
    latest_ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if latest_ckpt:
        print(f"🔁 Resuming from checkpoint: {latest_ckpt}")
        model.load_weights(latest_ckpt)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    print("\n✅ Training complete!")
    print(f"📦 Best model saved to: {checkpoint_path}")

    # Save final model
    final_model_path = os.path.join(CHECKPOINT_DIR, f"unet_final_{EXPERIMENT_NAME}.keras")
    model.save(final_model_path)
    print(f"💾 Final model exported to: {final_model_path}")


# ============================================================
# 🚦 ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
