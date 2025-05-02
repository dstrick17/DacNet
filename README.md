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
**Performance vs older models and publications on Test AUC scores per disease**
| Pathology           | original CheXNet | dannynet.py | vit_transformer.py | replicate_chexnet.py |
|---------------------|------------------|----------|------------------|--------------------|
| Atelectasis         | 0.8094           | **0.817** | 0.774           | 0.762              |
| Cardiomegaly        | 0.9248           | **0.932** | 0.89            | 0.922              |
| Consolidation       | **0.7901**        | 0.783    | 0.789           | 0.746              |
| Edema               | 0.8878           | **0.896** | 0.876           | 0.864              |
| Effusion            | 0.8638           | **0.905** | 0.857           | 0.883              |
| Emphysema           | 0.9371           | **0.963** | 0.828           | 0.85               |
| Fibrosis            | 0.8047           | **0.814** | 0.772           | 0.766              |
| Hernia              | 0.9164           | **0.997** | 0.872           | 0.925              |
| Infiltration        | **0.7345**        | 0.708    | 0.7             | 0.673              |
| Mass                | 0.8676           | **0.919** | 0.783           | 0.824              |
| Nodule              | 0.7802           | **0.789** | 0.673           | 0.646              |
| Pleural Thickening  | **0.8062**        | 0.801    | 0.766           | 0.756              |
| Pneumonia           | **0.768**         | 0.74     | 0.713           | 0.656              |
| Pneumothorax        | **0.8887**        | 0.875    | 0.821           | 0.827              |

---
### Average metrics across all diseases for each model
| Metric  | DannyNet | ViT Transformer | Replicate CheXNet |
|---------|----------|------------------|--------------------|
| Loss    | **0.0416** | 0.1589           | 0.1661             |
| AUC     | **0.8527** | 0.7940           | 0.7928             |
| F1      | **0.3861** | 0.1114           | 0.0763             |
---
### 📊 F1 Score Comparison for Each Model

| Disease             | DannyNet | ViT Transformer  | Replicate CheXNet |
|---------------------|----------|------------------|--------------------|
| **AVERAGE**         | **0.386** | 0.111           | 0.076              |
| Atelectasis         | **0.421** | 0.127           | 0.026              |
| Cardiomegaly        | **0.532** | 0.264           | 0.423              |
| Consolidation       | **0.226** | 0               | 0                  |
| Edema               | **0.286** | 0.004           | 0                  |
| Effusion            | **0.623** | 0.427           | 0.459              |
| Emphysema           | **0.516** | 0.079           | 0                  |
| Fibrosis            | **0.127** | 0               | 0                  |
| Hernia              | **0.750** | 0               | 0                  |
| Infiltration        | **0.395** | 0.193           | 0.061              |
| Mass                | **0.477** | 0.213           | 0.079              |
| Nodule              | **0.352** | 0.041           | 0                  |
| Pleural Thickening  | **0.258** | 0               | 0                  |
| Pneumonia           | **0.082** | 0               | 0                  |
| Pneumothorax        | **0.360** | 0.211           | 0.021              |


---
## Test-Images
Folder that contains labeled chest X-ray PNG files for the user to easily download and test on the Hugging Face Streamlit app.

---
## project_EDA.ipynb

Jupyter Notebook that conducts Exploratory Data Analysis such as how many different diseases are present in the dataset and the proportion of each disease in the dataset.
---

## Methodology
- **Data Preprocessing:** Resize, normalize, and augment X-ray images
- **Model Selection:** CNN and Transformer variants
- **Training & Validation:** Patient-level splits, loss monitoring
- **Evaluation:** Per-class AUC-ROC and F1 metrics
- **Comparison:** Benchmarks against original CheXNet results




