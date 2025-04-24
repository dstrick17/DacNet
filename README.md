# Chest X-Ray Classification Using Deep Learning  

## Test our model
Try out our model on Hugging Face Spaces:  
🔗 [DannyNet Demo](https://huggingface.co/spaces/cfgpp/Danny_Net_Demo)

## Project Overview  
This project replicates and extends the findings from the paper **[CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning](https://arxiv.org/abs/1711.05225)**.  
The goal is to evaluate how well convolutional neural networks (CNNs) can classify 14 chest pathologies using the NIH Chest X-ray dataset—and whether performance can approach or surpass that of expert radiologists.


## Project Structure 
Scripts
Data
Notebooks
.gitignore
README.MD

## Dataset  
- **Source:** [NIH Chest X-ray Dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data)  
- **Official NIH Release:** [NIH.gov Press Release](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community)  
- **Size:** 112,120 X-ray images from 30,805 patients  
- **Labels:** Multi-label classification of 14 thoracic diseases  
- **Preprocessing:**  
  - Resizing to 224x224  
  - Normalization (mean/std)  
  - Data augmentation (rotation, brightness, contrast, etc.)  
  - [ ] _Expand here: What libraries were used (e.g., `torchvision.transforms`)? Any oversampling?_

##  Models  
We experimented with the following deep learning architectures:

-  **DannyNet** (custom CNN using DenseNet-121 architecture, final model)  
-  **CheXNet Reimplementation** using DenseNet-121  
-  **vit_transformer** (attempt to use transformer models to improve AUC-ROC)    

> 📝 _Write detailed descriptions of each model here. Include diagrams or training configs if relevant._

---

## Methodology  
1. **Data Preprocessing:** Loading, resizing, and normalizing X-ray images  
2. **Model Selection:** Implementing a CNN architecture inspired by CheXNet  
3. **Training & Validation:** Splitting dataset, training with loss monitoring  
4. **Evaluation:** F1 score and AUC-ROC 
5. **Comparison:** Evaluating performance against the CheXNet study in disease classification  

## Installation & Setup  
### Clone the Repository  
```sh
git clone https://github.com/your-username/Deep-Learning-Project.git
cd Deep-Learning-Project
