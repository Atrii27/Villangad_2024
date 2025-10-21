import matplotlib.pyplot as plt
import numpy as np
import os

# === Results (replace with your actual final metrics if needed) ===
models = ["U-Net", "ResU-Net", "Attn U-Net", "Attn ResU-Net", "ASDMS U-Net"]
ious    = [0.5153, 0.6324, 0.8469, 0.9091, 0.9123]
accs    = [0.4457, 0.5508, 0.5931, 0.9574, 0.9602]

# === Output directory ===
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR,"raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed","test")
MODEL_DIR = os.path.join(ROOT, "experiments_all")
OUT_DIR = MODEL_DIR  # save confusion matrices here

# ==========================================================
# Figure 1: Grouped Bar Chart (IoU vs Accuracy)
# ==========================================================
x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(10,6))
plt.bar(x - width/2, ious, width, label='IoU', color='skyblue')
plt.bar(x + width/2, accs, width, label='Accuracy', color='salmon')

plt.title("📊 Final IoU & Accuracy Comparison — Villangad (2024)", fontsize=14, weight='bold')
plt.xlabel("Model", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.xticks(x, models, rotation=20)
plt.ylim(0, 1.05)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Annotate each bar with values
for i, (iou, acc) in enumerate(zip(ious, accs)):
    plt.text(i - 0.18, iou + 0.02, f"{iou:.2f}", color='black', fontsize=10)
    plt.text(i + 0.15, acc + 0.02, f"{acc:.2f}", color='black', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "metrics_bar_plot.png"), dpi=300)
plt.show()

print("✅ Saved bar chart to:", os.path.join(OUT_DIR, "metrics_bar_plot.png"))

# ==========================================================
# Figure 2: Line Chart (Performance Trend by Model Complexity)
# ==========================================================
plt.figure(figsize=(10,6))
plt.plot(models, ious, marker='o', linestyle='-', linewidth=2.5, color='royalblue', label='IoU')
plt.plot(models, accs, marker='s', linestyle='--', linewidth=2.5, color='darkorange', label='Accuracy')

plt.title("📈 Performance Trend Across Model Complexity — Villangad (2024)", fontsize=14, weight='bold')
plt.xlabel("Model Architecture", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.ylim(0, 1.05)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

# Annotate points
for i, (iou, acc) in enumerate(zip(ious, accs)):
    plt.text(models[i], iou + 0.02, f"{iou:.2f}", color='royalblue', ha='center', fontsize=10)
    plt.text(models[i], acc - 0.05, f"{acc:.2f}", color='darkorange', ha='center', fontsize=10)

plt.savefig(os.path.join(OUT_DIR, "metrics_trend_plot.png"), dpi=300)
plt.show()

print("✅ Saved line chart to:", os.path.join(OUT_DIR, "metrics_trend_plot.png"))
