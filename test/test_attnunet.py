# test/test_attnunet.py
import os, sys, numpy as np, tensorflow as tf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.attnunet import build_attnunet
from utils.dataset_loader import load_datasets
from utils.config import CHECKPOINT_DIR

def compute_metrics(y_true, y_pred):
    y_true = y_true.astype(bool); y_pred = y_pred.astype(bool)
    TP = np.logical_and(y_true, y_pred).sum()
    FP = np.logical_and(~y_true, y_pred).sum()
    FN = np.logical_and(y_true, ~y_pred).sum()
    TN = np.logical_and(~y_true, ~y_pred).sum()
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    iou = TP / (TP + FP + FN + 1e-8)
    dice = (2 * TP) / (2 * TP + FP + FN + 1e-8)
    f1 = (2 * precision * recall) / (precision + recall + 1e-8)
    acc = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    return precision, recall, iou, dice, f1, acc

_, _, test_ds = load_datasets()
X_test, y_test = [], []
for x, y in test_ds:
    X_test.append(x.numpy()); y_test.append(y.numpy())
X_test = np.concatenate(X_test, axis=0)
y_test = np.concatenate(y_test, axis=0)
print(f"✅ Test set loaded: {X_test.shape}")

model_path = os.path.join(CHECKPOINT_DIR, "attnunet_best.keras")
print(f"🧠 Loading model from: {model_path}")
model = tf.keras.models.load_model(model_path, compile=False)
print("✅ Model loaded successfully!")

y_pred = model.predict(X_test, verbose=1)
y_pred_bin = (y_pred > 0.5).astype(np.uint8)

precisions, recalls, ious, dices, f1s, accs = [], [], [], [], [], []
for i in range(len(y_test)):
    p, r, iou, d, f1, a = compute_metrics(y_test[i].squeeze(), y_pred_bin[i].squeeze())
    precisions.append(p); recalls.append(r); ious.append(iou)
    dices.append(d); f1s.append(f1); accs.append(a)

print("\n📊 Evaluation Results — Attention U-Net:")
print("------------------------------------------")
print(f"Precision : {np.mean(precisions):.4f}")
print(f"Recall    : {np.mean(recalls):.4f}")
print(f"IoU       : {np.mean(ious):.4f}")
print(f"Dice      : {np.mean(dices):.4f}")
print(f"F1 Score  : {np.mean(f1s):.4f}")
print(f"Accuracy  : {np.mean(accs):.4f}")
