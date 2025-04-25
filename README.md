# Chest X-Ray Classification Using Deep Learning

## Clone & Set Up
```sh
git clone https://github.com/your-username/Deep-Learning-Project.git
cd Deep-Learning-Project
pip install -r requirements.txt
```

---

## Try Our Model
Test our final model on [Hugging Face Spaces: DannyNet Demo](https://huggingface.co/spaces/cfgpp/Danny_Net_Demo)

---

## Running the Code
To train each model, navigate to the `scripts` directory and run the corresponding script. For example:
```sh
python scripts/dannynet.py
```

Evaluation results such as AUC and F1 scores will be printed in the console and logged to Weights & Biases (WandB) if your account is configured. The best model checkpoint will be saved in a `models/<run_id>` folder.

Example output screenshot:
![Results Preview](test-images/results_preview.png)  
*(Add this image manually to the test-images/ directory or update the path accordingly.)*

---

## Project Overview
This project replicates and extends the findings from the paper: [**CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning**](https://arxiv.org/abs/1711.05225).

We evaluate whether deep learning models, particularly CNNs and Transformers, can classify 14 chest pathologies from the NIH Chest X-ray dataset and potentially match or surpass expert radiologist performance.

---

## Project Structure
```text
Deep-Learning-Project/ 
  ├── eda.ipynb
  ├── scripts/ 
    ├── dannynet.py 
    ├── replicate_chexnet.py
    ├── vit_transformer.py 
  ├── XRay_app/ 
    ├── app.py
    ├── utils/
      ├── model_utils.py
      ├── preprocessing.py
  ├── test-images/
    ├── cardiomegaly.png
    ├── hernia-infiltration.png 
    ├── mass-nodule.png 
  ├── .gitignore 
  ├── requirements.txt 
  ├── README.md
```

---

## Dataset
- **Source:** [NIH Chest X-ray Dataset (Kaggle)](https://www.kaggle.com/datasets/nih-chest-xrays/data)
- **Official Release:** [NIH Press Release](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community)
- **Size:** 112,120 X-ray images from 30,805 unique patients
- **Labels:** Multi-label classification of 14 thoracic diseases

### Preprocessing
- Resizing to 224×224
- Normalization using ImageNet mean/std
- Data augmentation (e.g., rotation, brightness, contrast)
- Libraries used: `torchvision.transforms`
- Dataset filtering: Only includes images present in local folders and valid view positions (PA, AP)

---

## Models
We present three models that were trained as separate python scripts located in the scripts folder

### 1. `replicate_chexnet.py` – Baseline Reimplementation of CheXNet
Faithful reimplementation of the original CheXNet architecture using standard PyTorch tools.

**Key Features:**
- **Architecture:** DenseNet-121 (pretrained)
- **Classifier:** Fully connected output for 14 diseases
- **Loss:** BCEWithLogitsLoss (standard binary cross-entropy)
- **Optimizer:** Adam with `weight_decay=1e-5`
- **Scheduler:** ReduceLROnPlateau (patience=1)
- **Augmentations:**
  - `Resize(224)`
  - `RandomHorizontalFlip`
- **Evaluation:** AUC-ROC and F1 (threshold=0.5)
- **Filtering:** Uses only PA/AP views as per the original paper

**Training Strategy:**
- Patient-level splitting
- Early stopping (patience=5)
- Epochs: 20 max

**Hyperparameters:**
- Batch size: 16
- Learning rate: 0.001
- Seed: 42

**Purpose:**
Serves as a baseline for evaluating improvements from custom architectures like `dannynet.py`.

---

### 2. `dannynet.py` – Final Custom CNN Model (Best Performer)
A DenseNet-121-based CNN enhanced with Focal Loss and advanced augmentations to handle the class imbalance in the dataset.

**Key Features:**
- **Architecture:** DenseNet-121 (pretrained)
- **Classifier:** Fully connected layer outputting 14 logits
- **Loss:** Focal Loss (alpha=1, gamma=2)
- **Optimizer:** AdamW with `weight_decay=1e-5`
- **Scheduler:** ReduceLROnPlateau (patience=1, factor=0.1)
- **Augmentations:**
  - `RandomResizedCrop(224)`
  - `RandomHorizontalFlip`
  - `ColorJitter`
- **Evaluation:** AUC-ROC, F1 scores, optimal F1 thresholds per class

**Training Strategy:**
- Patient-level split to avoid data leakage
- Early stopping (patience=5)
- Epochs: 25 max

**Hyperparameters:**
- Batch size: 8
- Image size: 224x224
- Learning rate: 0.00005
- Seed: 42

**Performance:**
Highest average AUC and F1 across all models; used in final deployment/demo.

---

### 3. `vit_transformer.py` – Vision Transformer for Chest X-Ray Classification
Explores transformers for medical image classification using Hugging Face's `ViT-Base Patch16 224` model.

**Key Features:**
- **Architecture:** Vision Transformer (ViT) pretrained on ImageNet
- **Loss:** BCEWithLogitsLoss
- **Optimizer:** Adam (`lr=0.0001`, `weight_decay=1e-5`)
- **Scheduler:** ReduceLROnPlateau (patience=3)
- **Preprocessing:**
  - Uses `ViTFeatureExtractor`
  - Resize + normalize to transformer expectations

**Evaluation:**
- AUC-ROC and F1 (per class and average)
- Threshold = 0.5

**Training Strategy:**
- Patient-level split (80/10/10)
- 20 epochs max
- Early stopping (patience=5)

**Batch Size:** 16  
**Image Size:** 224x224

**Why Transformers?**
ViTs treat images as sequences of patches and apply self-attention to model global image features, which can be advantageous for complex medical images.

---

## Test-Images
Folder that contains 3 chest X-ray PNG files for the user to easily download and test on the Hugging Face Streamlit app.

---

## Methodology
- **Data Preprocessing:** Resize, normalize, and augment X-ray images
- **Model Selection:** CNN and Transformer variants
- **Training & Validation:** Patient-level splits, loss monitoring
- **Evaluation:** Per-class AUC-ROC and F1 metrics
- **Comparison:** Benchmarks against original CheXNet results




