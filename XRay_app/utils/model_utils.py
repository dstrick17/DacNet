# utils/model_utils.py
import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

# Disease labels
DISEASE_LIST = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

# Load trained CheXNet model
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

def load_model(model_path, device):
    model = CheXNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict(model, img_tensor, device):
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0).to(device))
        probs = torch.sigmoid(output[0]).cpu().numpy()
    return dict(zip(DISEASE_LIST, probs))



