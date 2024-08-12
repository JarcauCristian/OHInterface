import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class ImageDataset(Dataset):
    def __init__(self, dir: str):
        self.images = []
        self.labels = []

        for path in Path(dir).glob("*"):
            image = Image.open(path)
            self.images.append(transform(image))
            self.labels.append(14)

        # Convert lists to tensors
        self.images = torch.stack(self.images)  # Stack the list of images into a tensor
        self.labels = torch.tensor(self.labels)  # Convert the labels list to a tensor

    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.images[idx], self.labels[idx]
