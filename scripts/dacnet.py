import os
import argparse
import json
import random
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from tqdm.auto import tqdm
import wandb
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
import numpy as np
from torchvision.models import densenet121, DenseNet121_Weights
import time

CONFIG = {
    "model": "dannynet",
    "batch_size": 8,
    "learning_rate": 0.00005,
    "epochs": 25,
    "num_workers": 2,
    "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
    "data_dir": None,
    "wandb_project": "X-Ray Classification",
    "wandb_mode": "offline",
    "patience": 5,
    "seed": 42,
    "image_size": 224,
    "output_dir": "models",
    "max_train_batches": None,
    "max_eval_batches": None,
}

def discover_data_dir():
    candidates = [
        os.environ.get("NIH_DATA_DIR"),
        "/kaggle/input/data",
        "/kaggle/input/nih-chest-xrays/data",
        "/kaggle/input/nih-chest-xrays",
    ]
    for candidate in candidates:
        if candidate and os.path.exists(os.path.join(candidate, "Data_Entry_2017.csv")):
            return candidate
    kaggle_input = "/kaggle/input"
    if os.path.isdir(kaggle_input):
        for root, dirs, files in os.walk(kaggle_input):
            if "Data_Entry_2017.csv" in files and any(d.startswith("images_") for d in dirs):
                return root
    return None

def parse_args():
    parser = argparse.ArgumentParser(description="Train DACNet on NIH ChestX-ray14")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=discover_data_dir(),
        help="Path to the NIH ChestX-ray14 data directory containing Data_Entry_2017.csv and images_001 ... images_012"
    )
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"], help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=CONFIG["batch_size"], help="Batch size")
    parser.add_argument("--num_workers", type=int, default=CONFIG["num_workers"], help="DataLoader workers")
    parser.add_argument("--output_dir", default=CONFIG["output_dir"], help="Directory for checkpoints and results")
    parser.add_argument("--max_train_batches", type=int, default=CONFIG["max_train_batches"], help="Optional cap on training batches for smoke tests")
    parser.add_argument("--max_eval_batches", type=int, default=CONFIG["max_eval_batches"], help="Optional cap on validation/test batches for smoke tests")
    parser.add_argument(
        "--wandb_mode",
        default=CONFIG["wandb_mode"],
        choices=["online", "offline", "disabled"],
        help="Weights & Biases logging mode",
    )
    return parser.parse_args()

args = parse_args()
if not args.data_dir:
    raise ValueError("Provide --data_dir or set NIH_DATA_DIR to the NIH ChestX-ray14 directory.")

CONFIG.update({
    "data_dir": args.data_dir,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "num_workers": args.num_workers,
    "output_dir": args.output_dir,
    "wandb_mode": args.wandb_mode,
    "max_train_batches": args.max_train_batches,
    "max_eval_batches": args.max_eval_batches,
})

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG["seed"])

# Define image transformations (consistent with CheXNet)
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

 # Load and modify the model
model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
model.classifier = nn.Linear(model.classifier.in_features, 14)
model = model.to(CONFIG["device"])

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Define loss function and optimizer
criterion = FocalLoss(alpha=1, gamma=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-5) #Added weight decay. # betas=(0.9, 0.999) - this is default in pytorch
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.1)

# Load the CSV file with image metadata
data_path = os.path.abspath(CONFIG["data_dir"])

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data directory not found: {data_path}")

csv_file = os.path.join(data_path, "Data_Entry_2017.csv")

if not os.path.exists(csv_file):
    raise FileNotFoundError(f"Metadata file not found: {csv_file}")

print(f"Using dataset directory: {data_path}")

df = pd.read_csv(csv_file)

# Get list of all image folders from images_001 to images_012
image_folders = [os.path.join(data_path, f"images_{str(i).zfill(3)}", "images") for i in range(1, 13)]
# Create a dictionary mapping image filenames to their folder paths
image_to_folder = {}
for folder in image_folders:
    if os.path.exists(folder):
        for img_file in os.listdir(folder):
            if img_file.endswith('.png'):
                image_to_folder[img_file] = folder

# Filter the CSV to include only images that are present in the folders
df = df[df['Image Index'].isin(image_to_folder.keys())]
if df.empty:
    raise ValueError("No matching PNG images found. Expected images_001 ... images_012/images folders.")

# Unique patient IDs
unique_patients = df['Patient ID'].unique()

# Split patients — not rows
train_val_patients, test_patients = train_test_split(
unique_patients, test_size=0.02, random_state=CONFIG["seed"]
)

train_patients, val_patients = train_test_split(
train_val_patients, test_size=0.052, random_state=CONFIG["seed"]
)

#Use those patients to filter full image rows
train_df = df[df['Patient ID'].isin(train_patients)]
val_df   = df[df['Patient ID'].isin(val_patients)]
test_df  = df[df['Patient ID'].isin(test_patients)]


# List of diseases we’re classifying
disease_list = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

# Function to convert label string to a vector
def get_label_vector(labels_str):
    labels = labels_str.split('|')

    if labels == ['No Finding']:
        return [0] * len(disease_list)
    
    else:
        return [1 if disease in labels else 0 for disease in disease_list]
 
# Custom Dataset class
class CheXNetDataset(Dataset):
    def __init__(self, dataframe, image_to_folder, transform=None):
        self.dataframe = dataframe
        self.image_to_folder = image_to_folder
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['Image Index']
        folder = self.image_to_folder[img_name]

        img_path = os.path.join(folder, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        labels_str = self.dataframe.iloc[idx]['Finding Labels']
        label_vector = get_label_vector(labels_str)
        labels = torch.tensor(label_vector, dtype=torch.float)

        return image, labels

# Set up DataLoaders with our custom datasets
train_dataset = CheXNetDataset(train_df, image_to_folder, transform=transform_train)
val_dataset = CheXNetDataset(val_df, image_to_folder, transform=transform_test)
test_dataset = CheXNetDataset(test_df, image_to_folder, transform=transform_test)

trainloader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
valloader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
testloader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])


def get_optimal_thresholds(labels, preds):
    thresholds = []
    for i in range(preds.shape[1]):
        precision, recall, thresh = precision_recall_curve(labels[:, i], preds[:, i])
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_threshold = thresh[np.argmax(f1_scores)] if len(thresh) > 0 else 0.5
        thresholds.append(best_threshold)
    return thresholds

def evaluate(model, loader, criterion, device, desc="[Test]", thresholds=None):
    model.eval()
    running_loss = 0.0
    num_batches = 0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(loader, desc=desc)):
            if CONFIG["max_eval_batches"] is not None and batch_idx >= CONFIG["max_eval_batches"]:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            num_batches += 1
            preds = torch.sigmoid(outputs)
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    if not all_labels:
        raise ValueError("No evaluation batches were processed.")

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    if thresholds is None:
        thresholds = get_optimal_thresholds(all_labels, all_preds)

    preds_binary = np.zeros_like(all_preds)
    for i in range(all_preds.shape[1]):
        preds_binary[:, i] = (all_preds[:, i] > thresholds[i]).astype(int)

    auc_scores = [
        roc_auc_score(all_labels[:, i], all_preds[:, i])
        if len(np.unique(all_labels[:, i])) > 1 else float("nan")
        for i in range(14)
    ]
    f1_scores = [f1_score(all_labels[:, i], preds_binary[:, i]) for i in range(14)]

    avg_auc = np.nanmean(auc_scores)
    avg_f1 = np.mean(f1_scores)

    for i, disease in enumerate(disease_list):
        print(f"{desc} {disease} AUC: {auc_scores[i]:.4f} | F1: {f1_scores[i]:.4f}")

    print(f"{desc} Avg AUC: {avg_auc:.4f}, Avg F1: {avg_f1:.4f}")

    return {
        "loss": running_loss / num_batches,
        "avg_auc": avg_auc,
        "avg_f1": avg_f1,
        "auc_dict": dict(zip(disease_list, auc_scores)),
        "f1_dict": dict(zip(disease_list, f1_scores)),
        "thresholds": dict(zip(disease_list, thresholds))
    }


# Training function
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    device = CONFIG["device"]
    model.train()
    running_loss = 0.0
    num_batches = 0
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=True)
    for i, (inputs, labels) in enumerate(progress_bar):
        if CONFIG["max_train_batches"] is not None and i >= CONFIG["max_train_batches"]:
            break
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        num_batches += 1
        progress_bar.set_postfix({"loss": running_loss / num_batches})
    if num_batches == 0:
        raise ValueError("No training batches were processed.")
    train_loss = running_loss / num_batches
    return train_loss

def validate(model, valloader, criterion, device):
    return evaluate(model, valloader, criterion, device, desc="[Validate]")

 # Training loop with WandB and timestamped checkpoints
wandb.init(project=CONFIG["wandb_project"], config=CONFIG, mode=CONFIG["wandb_mode"])
wandb.watch(model, log="all")

transform_names = [t.__class__.__name__ for t in transform_train.transforms]

wandb.config.update({
    "model_architecture": "DenseNet121",
    "classifier_head": str(model.classifier),  # logs the Linear layer details
    "optimizer": optimizer.__class__.__name__,
    "loss_fn": criterion.__class__.__name__,
    "scheduler": scheduler.__class__.__name__,
    "augmentation": " + ".join(transform_names)
})


run_id = wandb.run.id
checkpoint_dir = os.path.join(CONFIG["output_dir"], run_id)
os.makedirs(checkpoint_dir, exist_ok=True)

best_val_auc = 0.0
patience_counter = 0
best_thresholds = None


for epoch in range(CONFIG["epochs"]):
    train_loss = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
    val_stats = validate(model, valloader, criterion, CONFIG["device"])
    scheduler.step(val_stats["loss"])

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_stats["loss"],
        "val_auc": val_stats["avg_auc"],
        "val_f1": val_stats["avg_f1"],
        "f1_dict": val_stats["f1_dict"],
        "auc_dict": val_stats["auc_dict"],
        "optimal_thresholds": val_stats["thresholds"],
})

    if best_thresholds is None or val_stats["avg_auc"] > best_val_auc:
        best_val_auc = val_stats["avg_auc"]
        best_thresholds = [val_stats["thresholds"][disease] for disease in disease_list]

        patience_counter = 0
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        checkpoint_path = os.path.join(checkpoint_dir, f"best_model_{timestamp}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        wandb.save(checkpoint_path)
    else:
        patience_counter += 1
        if patience_counter >= CONFIG["patience"]:
            print("Early stopping triggered.")
            break

# Evaluate the best model
best_checkpoint_path = sorted([os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.startswith('best_model_')])[-1]
model.load_state_dict(torch.load(best_checkpoint_path))
test_stats = evaluate(model, testloader, criterion, CONFIG["device"], thresholds=best_thresholds)
results = {
    "model": CONFIG["model"],
    "test_loss": test_stats["loss"],
    "test_auc": test_stats["avg_auc"],
    "test_f1": test_stats["avg_f1"],
    "test_auc_dict": test_stats["auc_dict"],
    "test_f1_dict": test_stats["f1_dict"],
    "thresholds": test_stats["thresholds"],
    "checkpoint": best_checkpoint_path,
    "config": CONFIG,
}
with open(os.path.join(checkpoint_dir, "test_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=float)

wandb.log({
    "test_loss": test_stats["loss"],
    "test_auc": test_stats["avg_auc"],
    "test_f1": test_stats["avg_f1"],
    "test_auc_dict": test_stats["auc_dict"],
    "test_f1_dict": test_stats["f1_dict"]
})

wandb.finish()
