# Villangad_2020 — 5 Supervised Deep Learning Models for Landslide Prediction

##  Overview
This repository implements **five advanced U-Net based supervised machine learning models**  
for **landslide detection and segmentation** using **PlanetScope Imagery (Villangad, 2024)**.  
The project integrates:

- Satellite image preprocessing and tiling
- Deep CNN segmentation architectures (U-Net variants)
- Training, validation, and testing pipelines
- Central image fine-tuning and visualization
- Confusion matrices and performance comparison
## Project Title

**5 Supervised Machine Learning Landslide Prediction Models on Area-Specific Landslides using PlanetScope Imagery (Villangad, 2024)**
**Worked Under the Supervision of:**  
**Dr. Yunus Ali Pulpadan**  
Assistant Professor, Earth and Environmental Science (EES), IISER Mohali  
**Developed by:**  
**Atrii Roy**  
3rd Year BS (Hons.) Data Science and Artificial Intelligence, IIT Guwahati  

##  Implemented Models
| Model | Description | Key Feature |
|-------|--------------|--------------|
| **U-Net** | Baseline CNN segmentation model | Encoder–decoder with skip connections |
| **ResU-Net** | Residual U-Net | Solves vanishing gradient, deeper learning |
| **Attention U-Net** | U-Net + Attention gates | Focuses on relevant landslide regions |
| **Attention ResU-Net** | Residual + Attention fusion | Improved precision and generalization |
| **ASDMS U-Net** | Attention + Dense Multi-Scale U-Net | Highest accuracy and robustness |

## Metrics & Results (Villangad, 2024)
| Model | IoU | Accuracy |
|--------|------|----------|
| U-Net | 0.5153 | 0.4457 |
| ResU-Net | 0.6324 | 0.5508 |
| Attention U-Net | 0.8469 | 0.5931 |
| Attention ResU-Net | 0.9091 | 0.9574 |
| ASDMS U-Net | 0.9123 | 0.9602 |
## Visual Outputs
- **Bar Chart:** `metrics_bar_plot.png`
- **Trend Line:** `metrics_trend_plot.png`
- **Confusion Matrices:** `confmat_<model>.png`
## Installation
# Clone the repository
git clone https://github.com/Atrii27/Villangad_2024.git
cd Villangad_2024
# Install dependencies
pip install -r requirements.txt
