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



from huggingface_hub import hf_hub_download
def load_model(device):
    model_path = hf_hub_download(repo_id="cfgpp/danny_net", filename="dannynet.pth")
    
    # Rebuild model architecture
    model = CheXNet(num_classes=14)
    
    # Load state dict (just weights)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    return model




def predict(model, img_tensor, device):
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0).to(device))
        probs = torch.sigmoid(output[0]).cpu().numpy()
    
    # Sort all results by descending probability
    sorted_probs = sorted(zip(DISEASE_LIST, probs), key=lambda x: x[1], reverse=True)
    return dict(sorted_probs)


import cv2
import numpy as np
import torch

def generate_gradcam(model, input_tensor, target_class, device):
    features = []
    gradients = []

    def forward_hook(module, input, output):
        features.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    last_conv_layer = model.features[-1]
    forward_handle = last_conv_layer.register_forward_hook(forward_hook)
    backward_handle = last_conv_layer.register_backward_hook(backward_hook)

    model.zero_grad()
    output = model(input_tensor.unsqueeze(0).to(device))
    class_score = output[0][target_class]
    class_score.backward()

    grads = gradients[0]
    fmap = features[0]

    weights = grads.mean(dim=[2, 3], keepdim=True)
    cam = (weights * fmap).sum(dim=1).squeeze()
    cam = torch.relu(cam).cpu().numpy()

    cam = cam - cam.min()
    cam = cam / cam.max()
    cam = cv2.resize(cam, (224, 224))

    forward_handle.remove()
    backward_handle.remove()

    return cam

    return dict(zip(DISEASE_LIST, probs))



