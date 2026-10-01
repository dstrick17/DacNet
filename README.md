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

If your dataset is stored elsewhere, set `NIH_DATA_DIR`:

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

### Kaggle GPU Smoke Test

For reviewers using Kaggle with the NIH ChestX-ray14 dataset attached and a GPU enabled, install the Kaggle-compatible requirements and run a short real-data smoke test:

```bash
pip install -r requirements-kaggle.txt
python scripts/dacnet.py \
  --data_dir /kaggle/input/datasets/organizations/nih-chest-xrays/data \
  --epochs 1 \
  --batch_size 4 \
  --num_workers 2 \
  --max_train_batches 5 \
  --max_eval_batches 2 \
  --wandb_mode offline
```

This checks the Kaggle dataset mount, GPU execution, training loop, evaluation loop, checkpointing, and `models/<run_id>/test_results.json` output without running a full training job.

---
## Running the Models

Once inside the Docker container, you can execute the training scripts.
 
### 1. Reproduction Baseline (CheXNet)

```bash
python scripts/replicate_chexnet.py --data_dir "$NIH_DATA_DIR"
```

To calculate secondary F1 scores from an existing trained checkpoint, select
one threshold per pathology on the validation set and apply those thresholds
unchanged to the held-out test set:

```bash
python scripts/replicate_chexnet.py \
  --data_dir "$NIH_DATA_DIR" \
  --evaluate_only \
  --checkpoint /path/to/best_model.pth \
  --output_dir models/chexnet-threshold-evaluation
```

This writes `thresholded_f1_results.json` containing both fixed-0.5 and
validation-thresholded F1 scores. This is a secondary analysis of the public
14-pathology test split, not a reproduction of the original expert-labeled F1
comparison because that expert-labeled test set is not publicly available.

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
- Sum of unweighted binary cross-entropy losses, matching the original
  paper's 14-pathology experiment
- Patient-level split
- AUROC as the primary reproduction metric
- Fixed-0.5 and validation-thresholded F1 as secondary metrics

### dacnet.py (DACNet)
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

## Verified Results

The final CheXNet reproduction and DACNet extension use the same patient-level
70%/10%/20% split: 21,563 training, 3,081 validation, and 6,161 test patients
(78,614/11,212/22,294 images). Machine-readable results and provenance are in
[`reproducibility/`](reproducibility/).

### Test AUROC per pathology

| Pathology | Original CheXNet | CheXNet reproduction | DACNet extension |
|---|---:|---:|---:|
| Atelectasis | 0.8094 | 0.7898 | 0.8281 |
| Cardiomegaly | 0.9248 | 0.9169 | 0.9134 |
| Consolidation | 0.7901 | 0.8003 | 0.8236 |
| Edema | 0.8878 | 0.8832 | 0.8952 |
| Effusion | 0.8638 | 0.8745 | 0.8831 |
| Emphysema | 0.9371 | 0.8769 | 0.9199 |
| Fibrosis | 0.8047 | 0.7824 | 0.8266 |
| Hernia | 0.9164 | 0.8471 | 0.9510 |
| Infiltration | 0.7345 | 0.6948 | 0.7122 |
| Mass | 0.8676 | 0.8161 | 0.8623 |
| Nodule | 0.7802 | 0.7219 | 0.7940 |
| Pleural Thickening | 0.8062 | 0.7813 | 0.8076 |
| Pneumonia | 0.7680 | 0.7386 | 0.7633 |
| Pneumothorax | 0.8887 | 0.8505 | 0.8839 |
| **Macro average** | Not reported | **0.8125** | **0.8474** |

### Secondary F1 analysis

One threshold per pathology was selected on the validation set and applied
unchanged to the held-out test set. This does not reproduce the original
expert-labeled pneumonia F1 comparison, whose test labels are not public.

| Pathology | CheXNet reproduction | DACNet extension |
|---|---:|---:|
| Atelectasis | 0.3654 | 0.4127 |
| Cardiomegaly | 0.4014 | 0.3942 |
| Consolidation | 0.2275 | 0.2445 |
| Edema | 0.2305 | 0.2396 |
| Effusion | 0.5158 | 0.5313 |
| Emphysema | 0.3626 | 0.4723 |
| Fibrosis | 0.1312 | 0.1650 |
| Hernia | 0.1026 | 0.4444 |
| Infiltration | 0.4038 | 0.4242 |
| Mass | 0.3200 | 0.3966 |
| Nodule | 0.2326 | 0.3211 |
| Pleural Thickening | 0.2004 | 0.2248 |
| Pneumonia | 0.0772 | 0.0863 |
| Pneumothorax | 0.3671 | 0.4218 |
| **Macro average** | **0.2813** | **0.3414** |

At a fixed threshold of 0.5, the CheXNet reproduction has a macro F1 of
0.1271. The validation-thresholded values above are the appropriate comparison
with DACNet's per-pathology threshold evaluation.

The Vision Transformer remains a supplementary experiment. Its historical
results are not included in this verified comparison because it has not been
rerun on the shared final split.


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
- Attach the final checkpoints and complete run bundles to the reviewed GitHub release.
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
