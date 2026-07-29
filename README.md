# DACNet: Reproduction and Extension of CheXNet for Chest X-ray Classification

This repository contains:
1. a **reproduction** of the CheXNet DenseNet-121 model on the NIH ChestX-ray14 dataset, and  
2. an **extension (DACNet)** designed to improve performance under class imbalance.

It also includes a Vision Transformer baseline and a Streamlit demo for inference.

---

## Reproduction Scope

This work reproduces key components of:

**CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning**  
Rajpurkar et al., 2017 (arXiv:1711.05225)

### What is reproduced
- DenseNet-121 architecture
- Multi-label classification (14 thoracic diseases)
- Patient-level dataset splitting
- Evaluation using AUC-ROC and F1 score

### Known differences from the original work
- No access to the original **expert-labeled test set**
- Labels are based on the publicly available NIH dataset
- Some implementation details are inferred due to lack of official code

---

## Repository Structure
```
DacNet/
├── scripts/
│   ├── replicate_chexnet.py
│   ├── dacnet.py
│   ├── vit_transformer.py
├── XRay_app/
├── reproducibility/
│   ├── DACNet/
│   ├── replicate_chexnet/
│   └── vit_transformer/
├── project_EDA.ipynb
├── requirements.txt
```
---

## Dataset

This project uses the **NIH ChestX-ray14 dataset**.

- ~112,000 frontal chest X-rays  
- 30,805 patients  
- 14 disease labels  

Download from:
https://www.kaggle.com/datasets/nih-chest-xrays/data

For cloud/Kaggle workflows that avoid downloading all images to a personal computer, see [DATA_ACCESS.md](DATA_ACCESS.md).

---

## Dataset Setup

Your directory must match what the scripts expect.

Example structure:

```
data/
├── Data_Entry_2017.csv
├── images_001/
├── images_002/
...
├── images_012/
```

If your dataset is stored elsewhere, update the path in the scripts:

```bash
export NIH_DATA_DIR=/path/to/nih_data
```

You can also pass the path directly with `--data_dir`.

---

## Reproducible Environment (Docker)

We provide a Docker environment to improve reproducibility.

### Build the container

```bash
docker build -t dacnet-env .
```

### Run the container

```bash
docker run --gpus all -it --rm -v "$(pwd)":/workspace/DacNet dacnet-env
```

Notes:
- Requires CUDA-compatible GPU  
- ≥8GB VRAM recommended  
- If no GPU, remove `--gpus all`  
- The image starts in `/workspace/DacNet` with WandB in offline mode by default.

---

## Quick Verification

Before running full training, verify the setup:

```bash
bash reproduce.sh
```

This checks:
- Python environment setup
- dependency installation
- app utility imports
If this step fails, resolve the Python environment before proceeding to full training.
---
## Running the Models

Once inside the Docker container, you can execute the training scripts.
 
### 1. Reproduction Baseline (CheXNet)

```bash
python scripts/replicate_chexnet.py --data_dir "$NIH_DATA_DIR"
```

### 2. DACNet (Improved CNN)

```bash
python scripts/dacnet.py --data_dir "$NIH_DATA_DIR"
```

### 3. Vision Transformer

```bash
python scripts/vit_transformer.py --data_dir "$NIH_DATA_DIR"
```

If you encounter CUDA memory errors, reduce batch size in the scripts.

---
 
Evaluation results (AUC and F1) print to the console and are saved as `models/<run_id>/test_results.json`. If WandB is configured, pass `--wandb_mode online`; otherwise scripts default to offline mode. The best model checkpoint is saved in the same `models/<run_id>` folder.

Useful reproducibility options:

```bash
python scripts/dacnet.py \
  --data_dir "$NIH_DATA_DIR" \
  --epochs 25 \
  --batch_size 8 \
  --output_dir models \
  --wandb_mode offline
```

---


## Models

### replicate_chexnet.py (Baseline)
DenseNet-121 reproduction of CheXNet.
- Multi-label classification (14 diseases)  
- BCEWithLogitsLoss  
- Patient-level split  
- Evaluated using AUC and F1  

### Dacnet.py (DACNet)
Improved CNN designed for class imbalance.
- DenseNet-121 backbone  
- Focal Loss  
- Per-class threshold tuning  
- Improved F1 performance  

### vit_transformer.py (ViT)
Transformer-based baseline.
- ViT-Base architecture  
- Multi-label classification  
- Compared against CNN approaches  

---

**Performance (Test AUC per Disease)**
| Pathology           | original CheXNet | Dacnet.py | vit_transformer.py | replicate_chexnet.py |
|---------------------|------------------|----------|-------------------|--------------------|
| Atelectasis         | 0.8094           | **0.817** | 0.774           | 0.762              |
| Cardiomegaly        | 0.9248           | **0.932** | 0.89            | 0.922              |
| Consolidation       | **0.7901**       | 0.783     | 0.789           | 0.746              |
| Edema               | 0.8878           | **0.896** | 0.876           | 0.864              |
| Effusion            | 0.8638           | **0.905** | 0.857           | 0.883              |
| Emphysema           | 0.9371           | **0.963** | 0.828           | 0.85               |
| Fibrosis            | 0.8047           | **0.814** | 0.772           | 0.766              |
| Hernia              | 0.9164           | **0.997** | 0.872           | 0.925              |
| Infiltration        | **0.7345**       | 0.708     | 0.7             | 0.673              |
| Mass                | 0.8676           | **0.919** | 0.783           | 0.824              |
| Nodule              | 0.7802           | **0.789** | 0.673           | 0.646              |
| Pleural Thickening  | **0.8062**       | 0.801     | 0.766           | 0.756              |
| Pneumonia           | **0.768**        | 0.74      | 0.713           | 0.656              |
| Pneumothorax        | **0.8887**       | 0.875     | 0.821           | 0.827              |

---
### Average metrics across all diseases for each model
| Metric  | DacNet | ViT Transformer | Replicate CheXNet |
|---------|----------|------------------|--------------------|
| Loss    | **0.0416** | 0.1589           | 0.1661             |
| AUC     | **0.8527** | 0.7940           | 0.7928             |
| F1      | **0.3861** | 0.1114           | 0.0763             |
---
### F1 Score Comparison for Each Model

| Disease             | DacNet | ViT Transformer  | Replicate CheXNet |
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

---

## Demo

Try the model here:  
https://huggingface.co/spaces/cfgpp/DACNet

Features:
- image upload  
- multi-label prediction  
- Grad-CAM visualization  

---

## Limitations

- Results rely on NIH labels (not expert annotations)  
- Some implementation details are inferred  
- Performance may vary across environments  
- The original CheXNet expert-labeled test set is not publicly available here
- Full training requires the NIH ChestX-ray14 data and GPU time

---

## ReScience C Readiness

This repository is public and includes open-source code, Docker setup, reproducibility metadata, result tables, figures, and reviewer-facing run commands. Before submission, complete the remaining journal artifacts:

- Add the ReScience C article source and metadata using the journal template.
- Archive the reviewed code release on Zenodo after acceptance to obtain a DOI.
- Archive any non-Kaggle data artifacts or generated outputs needed for exact verification.
- Use the Kaggle/cloud data access path in [DATA_ACCESS.md](DATA_ACCESS.md) for reviewer runs that should not require local image downloads.
- Confirm the submission is a replication of work by non-collaborating authors, as required by ReScience C.

---

## Citation

If you use this repository:

An Open-Source Reproduction and Enhancement of CheXNet for Chest X-ray Disease Classification  
https://arxiv.org/abs/2505.06646

---

## Acknowledgments

Rajpurkar et al.  
CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning  
https://arxiv.org/abs/1711.05225
