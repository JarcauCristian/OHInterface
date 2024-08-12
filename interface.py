import os
import torch
import shutil
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from ttkbootstrap import Style
from ttkbootstrap.constants import *
from torchvision import transforms, models
from foolbox import PyTorchModel, criteria
from foolbox.attacks import L2ProjectedGradientDescentAttack, LinfFastGradientAttack, L1ProjectedGradientDescentAttack
import eagerpy as ep

# Mapping from class index to text label for GTSRB
class_labels = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons"
}

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class ImageAttackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classifier and Attack Detector")
        self.root.geometry("800x600")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load models
        self.load_models()

        # Initialize GUI components
        self.initialize_gui()

    def load_models(self):
        # Load ResNet-18 model for classification
        self.model_resnet18 = models.resnet18(pretrained=False)
        self.model_resnet18.fc = torch.nn.Linear(self.model_resnet18.fc.in_features, 43)
        self.model_resnet18.load_state_dict(torch.load("resnet_gtsrb.pth"))
        self.model_resnet18 = self.model_resnet18.to(self.device).eval()

        # Load ResNet-50 model for attack detection
        self.model_resnet50 = models.resnet50(pretrained=False)
        self.model_resnet50.fc = torch.nn.Linear(self.model_resnet50.fc.in_features, 2)
        self.model_resnet50.load_state_dict(torch.load("model.pth"))
        self.model_resnet50 = self.model_resnet50.to(self.device).eval()

        # Wrap ResNet-18 model with Foolbox
        preprocessing = {'mean': torch.tensor([0.485, 0.456, 0.406]).to(self.device), 
                         'std': torch.tensor([0.229, 0.224, 0.225]).to(self.device), 
                         'axis': -3}
        self.fmodel_resnet18 = PyTorchModel(self.model_resnet18, bounds=(0, 1), preprocessing=preprocessing, device=self.device)

    def initialize_gui(self):
        self.style = Style(theme='darkly')

        frame = tk.Frame(self.root)
        frame.pack(pady=20)

        self.label_img = tk.Label(frame, text="No Image Loaded", bg="gray")
        self.label_img.pack(side=tk.TOP, pady=10)

        self.label_attacked_img = tk.Label(frame, text="Attacked Image", bg="gray")
        self.label_attacked_img.pack(side=tk.TOP, padx=10, pady=10)

        self.button_load = tk.Button(frame, text="Load Image", command=self.on_load_image)
        self.button_load.pack(side=tk.LEFT, padx=10, pady=10)

        self.button_attack = tk.Button(frame, text="Attack Image", command=self.on_attack_image)
        self.button_attack.pack(side=tk.LEFT, padx=10, pady=10)

        self.button_clear = tk.Button(frame, text="Clear", command=self.on_clear)
        self.button_clear.pack(side=tk.LEFT, padx=10, pady=10)

        self.label_pred_class = tk.Label(frame, text="Class: N/A")
        self.label_pred_class.pack(side=tk.TOP, pady=10)

        self.label_pred_attack = tk.Label(frame, text="Attacked: N/A")
        self.label_pred_attack.pack(side=tk.TOP, pady=10)

    def on_load_image(self):
        self.on_clear()

        self.filepath = filedialog.askopenfilename()
        if self.filepath:
            self.img = Image.open(self.filepath).convert('RGB')
            img_display = self.img.resize((200, 200), Image.Resampling.LANCZOS)
            img_display = ImageTk.PhotoImage(img_display)
            self.label_img.configure(image=img_display, text="")
            self.label_img.image = img_display

            # Make predictions
            self.predict_image(self.img)

    def on_clear(self):
        self.label_img.configure(image="", text="No Image Loaded")
        self.label_img.image = None
        self.label_attacked_img.configure(image="", text="Attacked Image")
        self.label_attacked_img.image = None
        self.label_pred_class.configure(text=f"Class: N/A")
        self.label_pred_attack.configure(text=f"Attacked: N/A")
        self.button_attack.config(state=tk.NORMAL)

    def predict_image(self, img):
        # Apply the transformation
        img_tensor = transform(img).unsqueeze(0).to(self.device)

        # Predict with ResNet-18 (classification)
        with torch.no_grad():
            prediction_class = self.model_resnet18(img_tensor)
            predicted_class = torch.argmax(prediction_class, dim=1).item()

        # Predict with ResNet-50 (attack detection)
        with torch.no_grad():
            prediction_attack = self.model_resnet50(img_tensor)
            predicted_attack = torch.argmax(prediction_attack, dim=1).item()

        # Convert class index to text label
        class_text = class_labels.get(predicted_class, "Unknown")

        self.label_pred_class.configure(text=f"Class: {class_text}")
        self.label_pred_attack.configure(text=f"Attacked: {'Yes' if predicted_attack == 1 else 'No'}")

        if predicted_attack == 1:
            self.button_attack.config(state=tk.DISABLED)
        else:
            self.button_attack.config(state=tk.NORMAL)

    def attack_image(self, img_tensor):
        # # Normalize after the attack is applied

        # # Ensure that the tensor is in the range [0, 1] before the attack
        # # Apply the attack
        
        # attack = LinfFastGradientAttack()
        # logits = self.model_resnet18(img_tensor.raw)
        # predictions = torch.argmax(logits, dim=1)

        # criterion = criteria.Misclassification(predictions)
        # _, clipped_advs, success = attack(self.fmodel_resnet18, img_tensor, criterion=criterion, epsilons=[0.1])

        # if success[0]:
        #     return clipped_advs[0][0].raw
        # else:
        #     return img_tensor

        img_tensor = img_tensor.raw

        # Ensure the image tensor requires gradients
        img_tensor.requires_grad = True

        # Set parameters for the PGD attack
        epsilon = 0.07  # Maximum perturbation
        alpha = 0.01  # Step size
        num_steps = 100  # Number of iterations

        # Save the original image for later projection
        img_tensor_orig = img_tensor.clone().detach()

        for _ in range(num_steps):
            # Zero all existing gradients
            self.model_resnet18.zero_grad()

            # Forward pass the data through the model
            outputs = self.model_resnet18(img_tensor)

            # Calculate the loss using the original predictions as target
            loss = torch.nn.functional.cross_entropy(outputs, outputs.max(1)[1])

            # Backward pass to calculate the gradients
            loss.backward()

            # Check if gradients are available
            if img_tensor.grad is None:
                raise RuntimeError("Gradient computation failed. Ensure the input tensor has requires_grad=True.")

            # Generate the adversarial image by taking a step in the direction of the gradient
            img_tensor = img_tensor + alpha * img_tensor.grad.sign()

            # Clip the perturbations to be within the epsilon ball
            perturbation = torch.clamp(img_tensor - img_tensor_orig, min=-epsilon, max=epsilon)
            img_tensor = torch.clamp(img_tensor_orig + perturbation, 0, 1)

            # Detach the image from the computation graph and re-enable gradient tracking
            img_tensor = img_tensor.detach()
            img_tensor.requires_grad = True

        return img_tensor.detach()

    def on_attack_image(self):
        if not hasattr(self, 'img') or self.img is None:
            return

        img_tensor = transform2(self.img).unsqueeze(0).to(self.device)
        img_tensor = ep.astensor(img_tensor)

        # Attack the image
        attacked_img_tensor = self.attack_image(img_tensor)

        # Convert the attacked tensor back to a PIL image
        attacked_img = transforms.ToPILImage()(attacked_img_tensor.squeeze(0))

        # Display the attacked image beside the original image
        img_display = attacked_img.resize((200, 200), Image.Resampling.LANCZOS)
        img_display = ImageTk.PhotoImage(img_display)
        self.label_attacked_img.configure(image=img_display, text="")
        self.label_attacked_img.image = img_display

        # Make predictions on attacked image
        self.predict_image(attacked_img)

def main():
    root = tk.Tk()
    app = ImageAttackerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
