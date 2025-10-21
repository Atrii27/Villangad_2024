# train/train_asdms_unet.py
import os, sys, datetime, tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.asdms_unet import build_asdms_unet
from utils.dataset_loader import load_datasets
from utils.config import LEARNING_RATE, EPOCHS, CHECKPOINT_DIR, LOG_DIR

def dice_bce_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1]); y_pred_f = tf.reshape(y_pred, [-1])
    inter = tf.reduce_sum(y_true_f * y_pred_f)
    dice_loss = 1 - (2.*inter + smooth)/(tf.reduce_sum(y_true_f)+tf.reduce_sum(y_pred_f)+smooth)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return dice_loss + bce

def main():
    train_ds, val_ds, _ = load_datasets()
    model = build_asdms_unet()

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss=dice_bce_loss,
                  metrics=["accuracy", tf.keras.metrics.MeanIoU(num_classes=2)])

    run_name = f"asdms_unet_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
    callbacks = [
        ModelCheckpoint(os.path.join(CHECKPOINT_DIR, f"{run_name}.keras"),
                        monitor="val_loss", save_best_only=True, verbose=1),
        CSVLogger(os.path.join(LOG_DIR, f"{run_name}.csv")),
        TensorBoard(log_dir=os.path.join(LOG_DIR, run_name)),
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=1)
    print(f"✅ ASDMS-UNet model saved to {CHECKPOINT_DIR}")

if __name__ == "__main__":
    main()
