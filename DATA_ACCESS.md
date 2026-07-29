# Data Access Without Local Image Downloads

DACNet uses the NIH ChestX-ray14 dataset, which is available from:

- Kaggle: https://www.kaggle.com/datasets/nih-chest-xrays/data
- Original NIH/Box link from the ChestX-ray8/ChestX-ray14 paper: https://nihcc.app.box.com/v/ChestXray-NIHCC

The full image set is large, so reviewers should not need to download it to a personal computer. The preferred paths are below.

## Option 1: Kaggle Notebook

Create a Kaggle Notebook, add the Kaggle dataset `nih-chest-xrays/data` as an input dataset, and upload or clone this repository into the notebook session.

Kaggle mounts input datasets under `/kaggle/input/...`. The training scripts auto-detect common Kaggle mount paths, so this is usually enough:

```bash
python scripts/dacnet.py --wandb_mode offline
```

If Kaggle mounts the dataset under a different folder, pass the detected folder explicitly:

```bash
python scripts/dacnet.py --data_dir /kaggle/input/data --wandb_mode offline
```

The directory passed to `--data_dir` must contain `Data_Entry_2017.csv` and the `images_001` through `images_012` folders.

## Option 2: Cloud VM or Colab Runtime

Use the Kaggle API or NIH/Box link inside the cloud runtime, and store the images on that runtime's attached disk or mounted cloud storage. This avoids downloading the dataset to a personal computer, but the cloud runtime still needs local filesystem access during training.

Example Kaggle API command in a cloud runtime:

```bash
kaggle datasets download -d nih-chest-xrays/data -p /workspace/nih_data --unzip
export NIH_DATA_DIR=/workspace/nih_data
python scripts/dacnet.py --wandb_mode offline
```

## Kaggle Python Environment

Kaggle notebooks may run a newer Python/PyTorch stack than the pinned local `requirements.txt`. On Kaggle, prefer the notebook's preinstalled GPU-compatible `torch` and `torchvision`, then install the Kaggle-specific requirements:

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

## Not Recommended: HTTP Streaming During Training

Training directly from remote HTTP URLs is possible in principle, but it is slower and less reliable for 112,000+ PNG files. For ReScience C review, a mounted Kaggle dataset or cloud disk is more reproducible.
