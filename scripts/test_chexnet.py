import torch
from torchvision import transforms
from PIL import Image
import argparse
import numpy as np
from torchvision.models import densenet121, DenseNet121_Weights
import torch.nn as nn

# Disease classes
DISEASE_LIST = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

# Image preprocessing
def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

# Model class (matches your training script)
class CheXNet(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        base_model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.features = base_model.features
        self.classifier = nn.Linear(base_model.classifier.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)

# Inference function
def predict(model, image_path, device):
    image = Image.open(image_path).convert('RGB')
    transform = get_transform()
    image_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]

    return dict(zip(DISEASE_LIST, probs))

# Run inference
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help="Path to input image")
    parser.add_argument('--model_path', type=str, required=True, help="Path to trained model .pth file")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CheXNet()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)

    results = predict(model, args.image_path, device)

    print("\nPredictions:")
    for disease, prob in results.items():
        print(f"{disease}: {prob:.4f}")


### How to run this code:  
# python test_chexnet.py /projectnb/dl4ds/projects/dca_project/nih_data/images_001/images/00000001_000.png --model_path /projectnb/dl4ds/projects/dca_project/scripts/models/59o12z7z-distinctive-snowflake-21/distinctive-snowflake-21.pth
