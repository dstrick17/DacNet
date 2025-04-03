# Build real neural network
import torch
import torch.nn as nn
import torch.nn.functional as F  # Fixed typo: funcitonal -> functional
import torch.optim as optim
import torchvision.transforms as transforms
import os
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, random_split, Dataset  # Added missing imports
from PIL import Image
import pandas as pd


## CheXNet is a 121-layer Dense Convolutional Network (DenseNet)

# Custom Dataset class for NIH ChestX-ray data
class ChestXrayDataset(Dataset):
    def __init__(self, data_dir, image_list_file, csv_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = pd.read_csv(csv_file)
        with open(image_list_file, 'r') as f:
            self.image_list = [line.strip() for line in f]
        self.data = self.data[self.data['Image Index'].isin(self.image_list)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['Image Index']
        folder = f"images_{img_name.split('-')[0][-3:]}/images"
        img_path = os.path.join(self.data_dir, folder, img_name)
        print(f"Attempting to open image: {img_path}")
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image not found at {img_path}")
            return None, None

        # Multi-label classification
        disease_list = ['Atelectasis', 'Cardiomegaly', 'Consolidation',
                        'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration',
                        'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
        labels = self.data.iloc[idx]['Finding Labels'].split('|')
        label_vector = torch.zeros(14, dtype=torch.float)

        if labels != ['No Finding']:
            for disease in labels:
                if disease in disease_list:
                    label_vector[disease_list.index(disease)] = 1

        if self.transform:
            image = self.transform(image)
        print(f"Loaded image: {img_name}, shape: {image.shape}, labels: {label_vector}")
        return image, label_vector


################################################################################
# Define a one epoch training function
################################################################################
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    """Train one epoch, e.g. all batches of one epoch."""
    device = CONFIG["device"]
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # put the trainloader iterator in a tqdm so it can print progress
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)

    # iterate through all batches of one epoch
    for i, (inputs, labels) in enumerate(progress_bar):
        print(f"Training batch {i}, inputs shape: {inputs.shape}, labels shape: {labels.shape}")
        inputs, labels = inputs.to(device), labels.to(device) # move inputs and labels to target device
        optimizer.zero_grad() #reset gradients to zero so they don't accumulate
        outputs = model(inputs) # Forward pass to get predictions
        loss = criterion(outputs, labels) # Calculate loss funciton (compare predictions with true labels)
        loss.backward() # Backpropagation
        optimizer.step() # Update weights - adjust parrmeters using computed gradients
        running_loss += loss.item()
        preds = torch.sigmoid(outputs)
        predicted = (preds > 0.5).float() # thresholding
        correct += (predicted == labels).sum().item()
        total += labels.numel()
        # Update progress_bar
        progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})

    train_loss = running_loss / len(trainloader) # calculate training loss
    train_acc = 100 * correct / total # calculate training accuracy -LOOK INTO USING AUC-ROC FOR MEDICAL IMAGING!!!
    return train_loss, train_acc

################################################################################
# Define a validation function
################################################################################
def validate(model, valloader, criterion, device):
    """Validate the model"""
    model.eval() # Set to evaluation
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): # Not tracking gradients but we can look into this ---------------------------
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False) # track progress of validation iterator
        # iterate through validation set
        for i, (inputs, labels) in enumerate(progress_bar):
            print(f"Validation batch {i}, inputs shape: {inputs.shape}, labels shape: {labels.shape}")
            inputs, labels = inputs.to(device), labels.to(device) # move inputs and labels to target device
            outputs = model(inputs) # Inference
            loss = criterion(outputs, labels) # Loss calculation
            running_loss += loss.item() # Add loss from this sample
            preds = torch.sigmoid(outputs)
            predicted = (preds > 0.5).float() # thresholding
            correct += (predicted == labels).sum().item()
            total += labels.numel()
            # Update progress bar
            progress_bar.set_postfix({"loss": running_loss / (i+1), "acc": 100. * correct / total}) 

        val_loss = running_loss/len(valloader) # Calculate validation loss
        val_acc = 100. * correct / total # Calculate validation accuracy
        return val_loss, val_acc #return validation loss and validation accuracy

############################################################################
#    Configuration Dictionary 
############################################################################
def main():
    CONFIG = {
        "model": "DenseNet",   # Change name when using a different model
        "batch_size": 8, # run batch size finder to find optimal batch size
        "learning_rate": 0.1,
        "epochs": 1,  # Train for longer in a real scenario
        "num_workers": 4, # Adjust based on your system
        "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./nih_data",  # Make sure this directory exists
        #"ood_dir": "./nih_data/ood-test",
        "wandb_project": "X-Ray Classification",
        "seed": 42,
    }

    import pprint
    print("\nCONFIG Dictionary: ")
    pprint.pprint(CONFIG)

##################################################################################
    #      Data Transformation 
#################################################################################
    # Training data transformations
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # Validatation Data and Test Data Transformations
    transform_val_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

#################################################################################
    #       Data Loading
##############################################################################


    # Load datasets
    data_path = CONFIG["data_dir"] # Retrieve data directory from a configuration dictionary
    train_val_csv = os.path.join(data_path, "Data_Entry_2017.csv") # construccts additional file to datapath
    train_val_list = os.path.join(data_path, "train_val_list.txt")
    test_list = os.path.join(data_path, "test_list.txt")

    trainset = ChestXrayDataset(data_path, train_val_list, train_val_csv, transform=transform_train)
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = random_split(trainset, [train_size, val_size])
    testset = ChestXrayDataset(data_path, test_list, train_val_csv, transform=transform_val_test)

    # Define loaders
    trainloader = DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    valloader = DataLoader(valset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    testloader = DataLoader(testset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    print(f"Trainloader length: {len(trainloader)}")
    print(f"Valloader length: {len(valloader)}")
    print(f"Testloader length: {len(testloader)}")


##################################################################################
    #   Instantiate model and move to target device
###################################################################################
    model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, 14) # Set the model to 14 classifiers
    # or any of these variants
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet169', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet161', pretrained=True)
    model.eval()
    model = model.to(CONFIG["device"]) # move model to target device

    print("\nModel summary:")
    print(f"{model}\n")

# Find optional batch size to run code -------------------------------------------
SEARCH_BATCH_SIZES = False # set this variable to True and run this code once to find the optimal batch size
if SEARCH_BATCH_SIZES:
    from utils import find_optimal_batch_size
    print("Finding optimal batch size...")
    optimal_batch_size = find_optimal_batch_size(model, trainset, CONFIG["device"], CONFIG["num_workers"])
    CONFIG["batch_size"] = optimal_batch_size
    print(f"Using batch size: {CONFIG['batch_size']}")
    

#################################################################################################################
    # Loss Function, Optimizer and optional learning rate scheduler
#####################################################################################
    criterion = nn.BCEWithLogitsLoss() #--- recommended over cross entropy for multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) 

    # Initialize wandb
    wandb.init(project="X-Ray Classification", config=CONFIG)
    wandb.watch(model)  # watch the model gradients

    ######################################################################################
        # --- Training Loop
    ######################################################################################
    best_val_acc = 0.0

    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        scheduler.step()

        # log to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"] # Log learning rate
        })

        # Save the best model (based on validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth") # Save to wandb as well

    wandb.finish()

    ######################################################################################
        # Evaluation
    ######################################################################################
    from sklearn.metrics import roc_auc_score
    import numpy as np

    def evaluate(model, testloader, criterion, device):
        model.eval()
        running_loss = 0.0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            progress_bar = tqdm(testloader, desc="[Test]", leave=False)
            for inputs, labels in progress_bar:
                print(f"Testing batch, inputs shape: {inputs.shape}, labels shape: {labels.shape}")
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                
                # Get probabilities with sigmoid (since BCEWithLogitsLoss uses logits)
                preds = torch.sigmoid(outputs)
                all_labels.append(labels.cpu())
                all_preds.append(preds.cpu())

        # Concatenate all predictions and labels
        all_labels = torch.cat(all_labels).numpy()
        all_preds = torch.cat(all_preds).numpy()

        # Calculate test loss
        test_loss = running_loss / len(testloader)

        # Calculate AUC-ROC for each class
        auc_scores = []
        for i in range(14):
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            auc_scores.append(auc)
        avg_auc = np.mean(auc_scores)

        print(f"Test Loss: {test_loss:.4f}, Average AUC-ROC: {avg_auc:.4f}")
        return test_loss, avg_auc

    # After training loop, load the best model and evaluate
    model.load_state_dict(torch.load("best_model.pth"))
    test_loss, test_auc = evaluate(model, testloader, criterion, CONFIG["device"])
    wandb.log({"test_loss": test_loss, "test_auc": test_auc})

if __name__ == "__main__":
    main()