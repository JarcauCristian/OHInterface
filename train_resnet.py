import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm

# Define paths
train_dir = './'
test_dir = './'
csv_file = './Train.csv'
test_csv_file = './Test.csv'
model_path = "resnet_gtsrb.pth"

# Hyperparameters
num_epochs = 5
batch_size = 32
learning_rate = 0.001
num_classes = 43  # 43 classes in the GTSRB dataset

# Data augmentation and normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Custom Dataset class
class GTSRBDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, f"{self.annotations.iloc[idx, -1]}")
        image = Image.open(img_name)
        label = int(self.annotations.iloc[idx, -2])  # Assuming the label is in the 8th column

        if self.transform:
            image = self.transform(image)

        return image, label

# Load datasets
train_dataset = GTSRBDataset(csv_file=csv_file, root_dir=train_dir, transform=data_transforms['train'])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = GTSRBDataset(csv_file=test_csv_file, root_dir=test_dir, transform=data_transforms['test'])
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load a pre-trained ResNet model and modify the final layer
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training loop
for epoch in tqdm(range(num_epochs)):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

print('Training complete')

# Testing loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in tqdm(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy on the test dataset: {accuracy:.2f}%')

# Save the trained model
torch.save(model.state_dict(), model_path)
print(f'Model saved to {model_path}')