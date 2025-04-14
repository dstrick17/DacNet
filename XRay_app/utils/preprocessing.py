from torchvision import transforms

def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Adjust if grayscale or RGB
    ])
    return transform(img)
