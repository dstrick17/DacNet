import torch
from torchvision import transforms
from PIL import Image
import argparse
import numpy as np
from torchvision.models import densenet121, DenseNet121_Weights
import torch.nn as nn
import os

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

# Model class (same as your training script)
class CheXNetWithDemographics(nn.Module):
    def __init__(self, base_model, demographic_dim=3, num_classes=14):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            base_model.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.demographic_layer = nn.Linear(demographic_dim, 128)
        self.classifier = nn.Sequential(
            nn.Linear(1024 + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, demographics):
        x = self.feature_extractor(x)
        d = self.demographic_layer(demographics)
        combined = torch.cat([x, d], dim=1)
        return self.classifier(combined)

# Inference function
def predict(model, image_path, age, gender, view, device):
    image = Image.open(image_path).convert('RGB')
    transform = get_transform()
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Normalize demographic data
    age = float(age) / 100.0
    gender = 0.0 if gender == 'M' else 1.0
    view_map = {'PA': 0.0, 'AP': 1.0}
    view = view_map.get(view, 0.5)

    demo_tensor = torch.tensor([[age, gender, view]], dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor, demo_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]

    return dict(zip(DISEASE_LIST, probs))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help="Path to the chest X-ray image")
    parser.add_argument('--age', type=float, required=True, help="Patient age (0–100)")
    parser.add_argument('--gender', type=str, required=True, choices=['M', 'F'], help="Patient gender")
    parser.add_argument('--view', type=str, required=True, choices=['PA', 'AP'], help="X-ray view position")
    parser.add_argument('--model_path', type=str, required=True, help="Path to model weights (.pth)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_densenet = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    model = CheXNetWithDemographics(base_densenet)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)

    results = predict(model, args.image_path, args.age, args.gender, args.view, device)

    print("\nPredictions:")
    for disease, prob in results.items():
        print(f"{disease}: {prob:.4f}")
