import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Path to the image and model
image_path = r"C:\Users\jarca_hqzqnz4\Downloads\00014_00004_00003.png"
model_path = "./resnet_gtsrb.pth"

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load and prepare the model
weights = models.ResNet18_Weights.IMAGENET1K_V1
model = models.resnet18(weights=weights)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 43)

# Load the saved model state dictionary
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Transformation to preprocess the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load and preprocess the image
image = Image.open(image_path)
image = transform(image).unsqueeze(0)  # Add batch dimension
image = image.to(device)

# Make prediction
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

# Print the output
print(f'Predicted label: {predicted.item()}')
