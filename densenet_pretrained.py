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

############################################################################
#    Configuration Dictionary 
############################################################################
CONFIG = {
    "model": "DenseNet",   # Change name when using a different model
    "batch_size": 8, # run batch size finder to find optimal batch size
    "learning_rate": 0.1,
    "epochs": 1,  # Train for longer in a real scenario
    "num_workers": 1, # Adjust based on your system
    "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
    "data_dir": "./nih_data",  # Make sure this directory exists
    #"ood_dir": "./nih_data/ood-test",
    "wandb_project": "X-Ray Classification",
    "seed": 42,
}
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
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=True)

    # iterate through all batches of one epoch
    for i, (inputs, labels) in enumerate(progress_bar):
        tqdm.write(f"Training batch {i}, inputs shape: {inputs.shape}, labels shape: {labels.shape}")
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
        progress_bar = tqdm(valloader, desc="[Validate]", leave=True) # track progress of validation iterator
        # iterate through validation set
        for i, (inputs, labels) in enumerate(progress_bar):
            tqdm.write(f"Validation batch {i}, inputs shape: {inputs.shape}, labels shape: {labels.shape}")
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



##################################################################################
    #      Custom Dataset class for NIH ChestX-ray data 
#################################################################################
# Make two list - one a list of all image indeces and images themselves.  one a list of all labels and image label 
# Get all images into one long list, then iterate through the list and corresponding label in .txt files
# for i in self.image list, load dataframe and column name which gets populated into self.labelslist
# 
# pROMPT gpt TO GET A LIST OF the entire path of Eevery single png file. Then match the file name of each png to the csv with the medadata. Then create a new list. One list for image paths, one list for labels.
# 
# Custom Dataset class


class ChestXrayDataset(Dataset):
    def __init__(self, data_dir, image_list_file, csv_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Load CSV with image labels
        df = pd.read_csv(csv_file)

        # ✅ Explicitly create lists for images and labels
        self.image_list = []
        self.labels_list = []

        # Read image filenames from text file
        with open(image_list_file, 'r') as f:
            image_filenames = [line.strip() for line in f]

        # Filter only relevant rows in the dataset
        filtered_df = df[df["Image Index"].isin(image_filenames)]

        # ✅ Populate the lists
        for _, row in filtered_df.iterrows():
            self.image_list.append(row["Image Index"])
            self.labels_list.append(row["Finding Labels"].split('|'))  # List of conditions

        print(f"Dataset initialized with {len(self.image_list)} images.")

        # Define disease labels as a class attribute
        self.disease_list = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
            'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
            'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
        ]

    def __getitem__(self, idx):
       
        img_name = self.image_list[idx]
        labels = self.labels_list[idx]  # Already a list

        # ✅ Adjust for NIH folder structure (e.g., "images_001")
        folder = f"images_{img_name[:3]}"  
        print(img_name, folder)
        exit()
        img_path = os.path.join(self.data_dir, folder, img_name)

        # # ✅ Handle missing images gracefully
        # if not os.path.exists(img_path):
        #     print(f"⚠️ Warning: Image {img_name} not found! Returning blank tensor.")
        #     image = torch.zeros((3, 224, 224))  # Placeholder tensor
        # else:
        #     try:
        #         image = Image.open(img_path).convert("RGB")
        #     except Exception as e:
        #         print(f"⚠️ Error loading {img_name}: {e}")
        #         image = torch.zeros((3, 224, 224))
        image = Image.open(img_path).convert("RGB")

        # ✅ Convert labels to binary vector
        label_vector = torch.zeros(len(self.disease_list))
        if "No Finding" not in labels:
            for label in labels:
                if label in self.disease_list:
                    label_vector[self.disease_list.index(label)] = 1

        # ✅ Apply transformation if provided
        if self.transform:
            print(image)
            image = self.transform(image)

        return image, label_vector

    def __len__(self):
        return len(self.image_list)


#################################################################################
    # Define Main
#################################################################################
def main():

    import pprint
    print("\nCONFIG Dictionary: ")
    pprint.pprint(CONFIG)



##################################################################################
########Test 
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    image_dir = "/projectnb/dl4ds/projects/dca_project/nih_data"
    # image_paths = [
    #     os.path.join(image_dir, fname)
    #     for fname in os.l(image_dir)
    # ]
    image_paths = []
    for root, dirs, files in os.walk(image_dir):
        for fname in files:
            # if os.path.splitext(fname)[1].lower() in ['png']:
            image_paths.append(os.path.join(root, fname))
    print(image_paths[:3])
    exit()

    dataset = ChestXrayDataset(
        data_dir="./nih_data",
        image_list_file="./nih_data/train_val_list.txt",
        csv_file="./nih_data/Data_Entry_2017.csv",
        transform=transform
    )

    # ✅ Check if dataset loads correctly
    for i in range(min(3, len(dataset))):
        image, labels = dataset[i]
        print(f"Sample {i}: Image Shape: {image.shape}, Labels: {labels}")

################################################################################
# Data transformations
################################################################################
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    transform_val_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


#################################################################################
    #       Data Loading
##############################################################################

    # Load datasets
    data_path = "./nih_data"
    train_val_csv = os.path.join(data_path, "Data_Entry_2017.csv")
    train_val_list = os.path.join(data_path, "train_val_list.txt")
    test_list = os.path.join(data_path, "test_list.txt")

    trainset = ChestXrayDataset(data_path, train_val_list, train_val_csv, transform=transform_train)
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = random_split(trainset, [train_size, val_size])
    testset = ChestXrayDataset(data_path, test_list, train_val_csv, transform=transform_val_test)

    # Define loaders
    trainloader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=1)
    valloader = DataLoader(valset, batch_size=8, shuffle=False, num_workers=1)
    testloader = DataLoader(testset, batch_size=8, shuffle=False, num_workers=1)

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
   # print(f"{model}\n")

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
            progress_bar = tqdm(testloader, desc="[Test]", leave=True)
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