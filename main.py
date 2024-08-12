# import os
# import pandas as pd
# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# from torch.utils.data import Dataset, DataLoader
# from torchvision.utils import save_image
# from foolbox import PyTorchModel, accuracy
# from foolbox.attacks import L1ProjectedGradientDescentAttack
# import torchvision.models as models
# import eagerpy as ep

# # Define transformations for the GTSRB dataset
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # ResNet models expect 224x224 input size
#     transforms.ToTensor(),
# ])

# # Custom dataset to handle the GTSRB test images
# class GTSRBDataset(Dataset):
#     def __init__(self, csv_file, root_dir, transform=None):
#         self.annotations = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.annotations)

#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, -1])
#         image = Image.open(img_name).convert("RGB")
#         label = int(self.annotations.iloc[idx, -2])  # Assuming label is in the 8th column
        
#         if self.transform:
#             image = self.transform(image)

#         return image, label

# # Load the GTSRB test dataset using the custom dataset class
# dataset_path = './'  # Replace with the path where GTSRB is extracted
# csv_file = os.path.join(dataset_path, 'Test.csv')
# test_dataset = GTSRBDataset(csv_file=csv_file, root_dir=dataset_path, transform=transform)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# # Step 1: Instantiate the model architecture (ResNet-18)
# model = models.resnet18(num_classes=43)  # Assuming the model was trained on 43 classes of GTSRB

# # Step 2: Load the state dictionary
# state_dict = torch.load('resnet_gtsrb.pth')  # Replace with your state_dict path

# # Step 3: Load the state dictionary into the model
# model.load_state_dict(state_dict)

# # Step 4: Set the model to evaluation mode
# model.eval()

# # Move model to GPU if available
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)

# # Wrap the model with Foolbox
# fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=None)

# # Load a batch of test images
# images, labels = next(iter(test_loader))
# images, labels = images.to(device), labels.to(device)

# # Instantiate the attack
# attack = L1ProjectedGradientDescentAttack()

# # Run the attack
# adversarials = attack(fmodel, images, labels, epsilons=[0.1])  # Unpack the tuple, keeping only the adversarial images

# # Convert adversarials to an EagerPy tensor if needed
# adversarials = ep.astensor(torch.stack(adversarials[0]).squeeze(1))

# # Check accuracy on adversarial examples
# acc = accuracy(fmodel, adversarials, labels)
# print(f"Accuracy on adversarial examples: {acc * 100:.1f} %")

# # Save the adversarial examples
# output_dir = './adversarial_images'  # Directory where images will be saved
# os.makedirs(output_dir, exist_ok=True)

# # Save each adversarial image (no denormalization needed since they're in [0, 1])
# for i, adv_img in enumerate(adversarials.tensor):
#     save_image(adv_img, os.path.join(output_dir, f'adversarial_{i}.png'))

# print(f"Adversarial images saved to {output_dir}")
import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from foolbox import PyTorchModel, accuracy
from foolbox.attacks import L1ProjectedGradientDescentAttack
import torchvision.models as models
import eagerpy as ep

# Define transformations for the GTSRB dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet models expect 224x224 input size
    transforms.ToTensor(),
])

# Custom dataset to handle the GTSRB test images
class GTSRBDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, -1])
        image = Image.open(img_name).convert("RGB")
        label = int(self.annotations.iloc[idx, -2])  # Assuming label is in the 8th column
        
        if self.transform:
            image = self.transform(image)

        return image, label

# Load the GTSRB test dataset using the custom dataset class
dataset_path = './'  # Replace with the path where GTSRB is extracted
csv_file = os.path.join(dataset_path, 'Test.csv')
test_dataset = GTSRBDataset(csv_file=csv_file, root_dir=dataset_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Step 1: Instantiate the model architecture (ResNet-18)
model = models.resnet18(num_classes=43)  # Assuming the model was trained on 43 classes of GTSRB

# Step 2: Load the state dictionary
state_dict = torch.load('resnet_gtsrb.pth')  # Replace with your state_dict path

# Step 3: Load the state dictionary into the model
model.load_state_dict(state_dict)

# Step 4: Set the model to evaluation mode
model.eval()

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Wrap the model with Foolbox
fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=None)

# Load a batch of test images
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

# Instantiate the attack
attack = L1ProjectedGradientDescentAttack()

# Run the attack
adversarials_tuple = attack(fmodel, images, labels, epsilons=[0.1])

# Assuming the first element of the tuple is a list of tensors for each image in the batch
adversarials = adversarials_tuple[0]  # Extract the list of adversarial examples

# Convert the list of tensors to a single tensor
adversarials = torch.stack(adversarials)

# Ensure the tensor has the correct shape (remove any unnecessary dimensions)
adversarials = adversarials.squeeze()

# If the shape is still incorrect, flatten to 4D [batch_size, 3, 224, 224]
if adversarials.ndimension() == 5 and adversarials.size(0) == 1:
    adversarials = adversarials.squeeze(0)

# Convert to EagerPy tensor if needed
adversarials = ep.astensor(adversarials)

# Check accuracy on adversarial examples
acc = accuracy(fmodel, adversarials, labels)
print(f"Accuracy on adversarial examples: {acc * 100:.1f} %")

# Save the adversarial examples
output_dir = './adversarial_images'  # Directory where images will be saved
os.makedirs(output_dir, exist_ok=True)

# Save each adversarial image (no denormalization needed since they're in [0, 1])
for i, adv_img in enumerate(adversarials.raw):
    save_image(adv_img, os.path.join(output_dir, f'adversarial_{i}.png'))

print(f"Adversarial images saved to {output_dir}")
