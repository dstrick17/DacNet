import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from tqdm.auto import tqdm
import wandb
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np

# Configuration settings
CONFIG = {
    "model": "DenseNet_pretrained",
    "batch_size": 128,
    "learning_rate": 0.01,
    "epochs": 10,
    "num_workers": 8,
    "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
    "data_dir": "/projectnb/dl4ds/projects/dca_project/nih_data",
    "wandb_project": "X-Ray Classification",
    "seed": 42,
}

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

# Split the data into train+val and test (80% train+val, 20% test)
train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=CONFIG["seed"])

# Further split train+val into train and val (60% train, 20% val of total)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=CONFIG["seed"])  # 0.25 ≈ 20/80

# Create lists of indices for each set
train_indices = train_df.index.tolist()
val_indices = val_df.index.tolist()
test_indices = test_df.index.tolist()

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

# Function to load images and labels for a batch of indices
def collate_fn(batch_indices, df, image_to_folder, transform):
    images = []
    labels = []
    
    for idx in batch_indices:
        img_name = df.loc[idx, 'Image Index']
        folder = image_to_folder[img_name]
        img_path = os.path.join(folder, img_name)

        try:
            image = Image.open(img_path).convert('RGB')
            image = transform(image)

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = torch.zeros((3, 224, 224))  # Use blank image if loading fails

        labels_str = df.loc[idx, 'Finding Labels']
        label_vector = get_label_vector(labels_str)

        images.append(image)
        labels.append(label_vector)

    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.float)

    return images, labels

# Set up DataLoaders with our lists of indices
trainloader = DataLoader(train_indices, batch_size=CONFIG["batch_size"], shuffle=True, 
                         num_workers=CONFIG["num_workers"], 
                         collate_fn=lambda x: collate_fn(x, df, image_to_folder, transform), persistent_workers=True)
valloader = DataLoader(val_indices, batch_size=CONFIG["batch_size"], shuffle=False, 
                       num_workers=CONFIG["num_workers"], 
                       collate_fn=lambda x: collate_fn(x, df, image_to_folder, transform))
testloader = DataLoader(test_indices, batch_size=CONFIG["batch_size"], shuffle=False, 
                        num_workers=CONFIG["num_workers"], 
                        collate_fn=lambda x: collate_fn(x, df, image_to_folder, transform))

# Evaluation function
def evaluate(model, testloader, criterion, device, desc="[Test]"):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        progress_bar = tqdm(
            trainloader,
            desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]",
            position=0,
            leave=True,
            ascii=True,
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
            )

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.cuda.amp.autocast(enabled=device.startswith('cuda')):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            preds = torch.sigmoid(outputs)

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()

    test_loss = running_loss / len(testloader)
    auc_scores = [roc_auc_score(all_labels[:, i], all_preds[:, i]) for i in range(14)]
    avg_auc = np.mean(auc_scores)

    # F1 score: binarize predictions at 0.5 threshold
    preds_binary = (all_preds > 0.5).astype(int)

    f1 = f1_score(all_labels, preds_binary, average='macro')  # Macro-average for multi-label
    print(f"{desc} Loss: {test_loss:.4f}, Avg AUC-ROC: {avg_auc:.4f}, F1 Score: {f1:.4f}")
    return test_loss, avg_auc, f1

# Load and modify the model
model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, 14)  # 14 disease classes
model = model.to(CONFIG["device"])

# Define loss function, optimizer, scheduler, scaler
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=1e-6)
scaler = torch.cuda.amp.GradScaler(enabled=CONFIG["device"] in ('cuda', 'mps'))

# Training function
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    device = CONFIG["device"]
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
            trainloader,
            desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]",
            position=0,
            leave=True,
            ascii=True,
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
            )

    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=CONFIG["device"].startswith('cuda')):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        preds = torch.sigmoid(outputs)
        predicted = (preds > 0.5).float()

        correct += (predicted == labels).sum().item()
        total += labels.numel()

        progress_bar.set_postfix({
            "loss": f"{running_loss/(i+1):.4f}",
            "pos_acc": f"{(100.*correct/total):.1f}%"
            })

    train_loss = running_loss / len(trainloader)
    train_acc = 100 * correct / total

    return train_loss, train_acc

# Validation function
def validate(model, valloader, criterion, device):

    val_loss, val_auc, val_f1 = evaluate(model, valloader, criterion, device, desc="[Validate]")
    return val_loss, val_auc, val_f1

# Start training with Weights & Biases logging
wandb.init(project=CONFIG["wandb_project"], config=CONFIG)
wandb.watch(model, log= "all", log_freq=100)

best_val_auc = 0.0  # Switch to tracking best AUC instead of accuracy
for epoch in range(CONFIG["epochs"]):
    train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
    val_loss, val_auc, val_f1 = validate(model, valloader, criterion, CONFIG["device"])

    scheduler.step()

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_auc": val_auc,
        "val_f1": val_f1,
        "learning_rate": optimizer.param_groups[0]['lr']
    })

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_auc': best_val_auc,
        }, "best_model.pth")

        wandb.save("best_model.pth")
        print(f"New best model saved with AUC: {val_auc:.4f}")
    
    wandb.save("best_model.pth")
    print(f"New best model saved with AUC: {val_auc:.4f}")

# Evaluate the best model
checkpoint = torch.load("best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded best model from epoch {checkpoint['epoch']} with AUC {checkpoint['best_auc']:.4f}")

test_loss, test_auc, test_f1 = evaluate(model, testloader, criterion, CONFIG["device"])

wandb.log({"test_loss": test_loss, "test_auc": test_auc, "test_f1": test_f1})
wandb.finish()
