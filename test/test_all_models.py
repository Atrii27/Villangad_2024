import os
import numpy as np
import pandas as pd
from glob import glob
from utils.dataset_loader import load_datasets
from models.unet import build_unet
from models.resunet import build_resunet
from models.attnunet import build_attn_unet
from models.attnresunet import build_attn_resunet
from models.asdms_unet import build_asdms_unet
from utils.config import OUTPUTS_DIR

# --- Evaluation metrics ---
def iou(y_true, y_pred, t=0.5):
    y_pred = (y_pred > t).astype(np.uint8)
    inter = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return inter / (union + 1e-8)

def prec_recall_f1(y_true, y_pred, t=0.5):
    y_pred = (y_pred > t).astype(np.uint8)
    tp = np.logical_and(y_true == 1, y_pred == 1).sum()
    fp = np.logical_and(y_true == 0, y_pred == 1).sum()
    fn = np.logical_and(y_true == 1, y_pred == 0).sum()
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return p, r, f1

# --- Model mapping ---
MODELS = {
    "unet": build_unet,
    "resunet": build_resunet,
    "attnunet": build_attn_unet,
    "attnresunet": build_attn_resunet,
    "asdms_unet": build_asdms_unet,
}

# --- Checkpoint lookup ---
CHECKPOINTS = {
    name: glob(os.path.join(OUTPUTS_DIR, "checkpoints", name, "*.keras"))[0]
    for name in MODELS.keys()
}

# --- Evaluate ---
def evaluate_model(name, model_fn, checkpoint, test_ds):
    print(f"\n🔍 Evaluating {name} ...")
    model = model_fn()
    model.load_weights(checkpoint)
    ious, f1s, ps, rs = [], [], [], []
    for imgs, msks in test_ds:
        preds = model.predict(imgs, verbose=0)
        for i in range(len(preds)):
            y_true = msks[i].numpy().squeeze()
            y_pred = preds[i].squeeze()
            ious.append(iou(y_true, y_pred))
            p, r, f1 = prec_recall_f1(y_true, y_pred)
            ps.append(p); rs.append(r); f1s.append(f1)
    return {
        "Model": name,
        "IoU": np.mean(ious),
        "Precision": np.mean(ps),
        "Recall": np.mean(rs),
        "F1": np.mean(f1s),
    }

def main():
    _, _, test_ds = load_datasets(batch=1)
    results = []

    for name, fn in MODELS.items():
        ckpt = CHECKPOINTS.get(name)
        if not ckpt:
            print(f"⚠️ No checkpoint found for {name}")
            continue
        results.append(evaluate_model(name, fn, ckpt, test_ds))

    df = pd.DataFrame(results)
    log_dir = os.path.join(OUTPUTS_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Save TXT
    txt_path = os.path.join(log_dir, "all_model_metrics.txt")
    with open(txt_path, "w") as f:
        f.write("📊 Combined Model Evaluation Results\n")
        f.write("=" * 45 + "\n\n")
        for _, row in df.iterrows():
            f.write(f"Model: {row['Model']}\n")
            f.write(f"  IoU       : {row['IoU']:.4f}\n")
            f.write(f"  Precision : {row['Precision']:.4f}\n")
            f.write(f"  Recall    : {row['Recall']:.4f}\n")
            f.write(f"  F1 Score  : {row['F1']:.4f}\n")
            f.write("-" * 45 + "\n")

    print("\n✅ Results saved:")
    print(f"TXT: {txt_path}")
    print("\n📄 Summary:\n", df)

if __name__ == "__main__":
    main()
