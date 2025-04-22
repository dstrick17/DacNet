# enabled Wandb to track scheduler, optimizer, model, augmentation
# editied preds_binary to check different F1 thresholds
######## source ~/chexnet310/bin/activate

import os
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
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import time
import math

CONFIG = {
    "model": "danny_net",
    "batch_size": 32,
    "learning_rate": 0.00005,
    "epochs": 9,
    "num_workers": 5,
    "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
    "data_dir": "/projectnb/dl4ds/projects/dca_project/nih_data",
    "wandb_project": "X-Ray Classification",
    "patience": 5,
    "seed": 42,
    "image_size": 224,
    "transformer_hidden_dim": 512,
    "transformer_heads": 8,
    "transformer_layers": 2,
}

# Define image transformations (consistent with CheXNet)
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet normalization
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class DenseNetTransformer(nn.Module):
    def __init__(self, num_classes=14, hidden_dim=512, nhead=8, num_layers=2):
        super().__init__()
        base_model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        
        # Correct feature extraction - using features up to the first transition block
        self.feature_extractor = nn.Sequential(
            base_model.features[:5],  # Outputs 256 channels at 56x56
            nn.Conv2d(256, hidden_dim, kernel_size=1)  # Updated to expect 256 channels
        )
        
        # Positional encoding for spatial information
        self.pos_encoder = PositionalEncoding2D(hidden_dim)
        
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Extract features with proper spatial resolution
        x = self.feature_extractor(x)  # [B, 512, 56, 56]
        
        # Adaptive average pool to get to 28x28 if needed
        x = nn.AdaptiveAvgPool2d(28)(x)  # [B, 512, 28, 28]
        
        x = x.flatten(2).permute(0, 2, 1)  # [B, 784, 512]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer processing
        x = self.transformer(x)
        
        # Global average pooling and classification
        x = x.mean(dim=1)
        return self.classifier(x)

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(1, 28*28, d_model)
        position = torch.arange(0, 28*28, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

 # Load and modify the model
model = DenseNetTransformer(num_classes=14).to(CONFIG["device"])
model = model.to(CONFIG["device"])


# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-5) #Added weight decay. # betas=(0.9, 0.999) - this is default in pytorch
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.1)

# Load the CSV file with image metadata
data_path = CONFIG["data_dir"]
csv_file = os.path.join(data_path, "Data_Entry_2017.csv")
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

def evaluate(model, loader, criterion, device, desc="[Test]"):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=desc):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            preds = torch.sigmoid(outputs)
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    thresholds = get_optimal_thresholds(all_labels, all_preds)

    preds_binary = np.zeros_like(all_preds)
    for i in range(all_preds.shape[1]):
        preds_binary[:, i] = (all_preds[:, i] > thresholds[i]).astype(int)

    auc_scores = [roc_auc_score(all_labels[:, i], all_preds[:, i]) for i in range(14)]
    f1_scores = [f1_score(all_labels[:, i], preds_binary[:, i]) for i in range(14)]

    avg_auc = np.mean(auc_scores)
    avg_f1 = np.mean(f1_scores)

    for i, disease in enumerate(disease_list):
        print(f"{desc} {disease} AUC: {auc_scores[i]:.4f} | F1: {f1_scores[i]:.4f}")

    print(f"{desc} Avg AUC: {avg_auc:.4f}, Avg F1: {avg_f1:.4f}")

    return {
        "loss": running_loss / len(loader),
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
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=True)
    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix({"loss": running_loss / (i + 1)})
    train_loss = running_loss / len(trainloader)
    return train_loss

def validate(model, valloader, criterion, device):
    return evaluate(model, valloader, criterion, device, desc="[Validate]")

 # Training loop with WandB and timestamped checkpoints
wandb.init(project=CONFIG["wandb_project"], config=CONFIG)
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
checkpoint_dir = os.path.join("models", run_id)
os.makedirs(checkpoint_dir, exist_ok=True)

best_val_auc = 0.0
patience_counter = 0


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

    if val_stats["avg_auc"] > best_val_auc:
        best_val_auc = val_stats["avg_auc"]

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
test_stats = evaluate(model, testloader, criterion, CONFIG["device"])
wandb.log({
    "test_loss": test_stats["loss"],
    "test_auc": test_stats["avg_auc"],
    "test_f1": test_stats["avg_f1"],
    "test_auc_dict": test_stats["auc_dict"],
    "test_f1_dict": test_stats["f1_dict"]
})

wandb.finish()
