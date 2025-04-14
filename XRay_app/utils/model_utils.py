import torch
import torch.nn as nn

class DummyModel(nn.Module):
    def __init__(self, num_classes=2):
        super(DummyModel, self).__init__()
        self.num_classes = num_classes

    def forward(self, x):
        batch_size = x.shape[0]
        return torch.tensor([[2.0, 1.0]] * batch_size)  # fixed logits

def load_model(path=None, model_name=None):
    return DummyModel()

def predict(model, img_tensor):
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0))
        probs = torch.nn.functional.softmax(output[0], dim=0)
    return probs


