# Villangad_2020 — 5 Supervised Deep Learning Models for Landslide Prediction
##  Overview
This repository implements **five advanced U-Net-based supervised deep learning models** for **landslide detection and segmentation** using **PlanetScope imagery (Villangad, 2024)**.
It covers:
- Satellite image preprocessing & tiling  
- CNN segmentation (U-Net family)  
- Training / validation / testing pipelines  
- Central image prediction & visualization  
- Confusion matrices, metrics, and model comparison  
---
## Project Title
**5 Supervised Machine Learning Landslide Prediction Models on Area-Specific Landslides using PlanetScope Imagery (Villangad, 2024)**  
**Supervised by:**  
*Dr. Yunus Ali Pulpadan* — Assistant Professor, Earth and Environmental Science (EES), IISER Mohali  
**Developed by:**  
*Atrii Roy* — 3rd Year BS (Hons.) Data Science & Artificial Intelligence, IIT Guwahati  
---
##  Implemented Models
| Model | Description | Key Feature |
|--------|--------------|-------------|
| **U-Net** | Baseline CNN segmentation model | Encoder–decoder with skip connections |
| **ResU-Net** | Residual U-Net | Deeper learning, vanishing-gradient solution |
| **Attention U-Net** | U-Net + Attention Gates | Focuses on relevant landslide regions |
| **Attention ResU-Net** | Residual + Attention Fusion | Improves precision & contextual learning |
| **ASDMS U-Net** | Attention + Dense Multi-Scale | Highest robustness & accuracy |

---
## 📊 Metrics & Results (Villangad, 2024)
| Model | IoU | Accuracy |
|--------|------|----------|
| **U-Net** | 0.5153 | 0.4457 |
| **ResU-Net** | 0.6324 | 0.5508 |
| **Attention U-Net** | 0.8469 | 0.5931 |
| **Attention ResU-Net** | 0.9091 | 0.9574 |
| **ASDMS U-Net** | 0.9123 | 0.9602 |

> **Observation:** Attention-based and multi-scale models outperformed the baseline U-Net, achieving the best recall and IoU scores.
---
## Visual Outputs
- **Bar Chart:** `metrics_bar_plot.png`  
- **Confusion Matrices:** `confmat_<model>.png`  
- **Predicted Maps:** `prediction_<model>.png`  
- **Radar & Trend Plots:** `model_metrics_radar.png`
---
## Installation
# Clone the repository
git clone https://github.com/Atrii27/Villangad_2020.git
cd Villangad_2020
# Install dependencies
pip install -r requirements.txt

