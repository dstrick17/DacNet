# Pneumonia Detection Using Deep Learning  

## Project Overview  
This project aims to reproduce and adapt the findings of the paper **"CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning"** by training a convolutional neural network (CNN) on a publicly available pneumonia dataset from Kaggle. The goal is to evaluate whether deep learning can match or surpass radiologists in pneumonia detection.  

## Project Structure 


## Dataset  
- **Source:** [Kaggle Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) 
- ** Link to NIH paper:**  (https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community)
- **Classes:** Binary classification (Normal vs. Pneumonia)  
- **Total Images:** 5,863 (1,583 Normal, 4,280 Pneumonia)  
- **Preprocessing Steps:** Resizing, normalization, augmentation (to be detailed)  

## Methodology  
1. **Data Preprocessing:** Loading, resizing, and normalizing X-ray images  
2. **Model Selection:** Implementing a CNN architecture inspired by CheXNet  
3. **Training & Validation:** Splitting dataset, training with loss monitoring  
4. **Evaluation:** Computing accuracy, F1 score, AUC-ROC, and confusion matrix  
5. **Comparison:** Evaluating performance against the CheXNet study  

## Installation & Setup  
### Clone the Repository  
```sh
git clone https://github.com/your-username/Deep-Learning-Project.git
cd Deep-Learning-Project
