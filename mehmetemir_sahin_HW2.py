import os
import random
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# New dataset with rotated MNIST images
class RotatedMNIST(Dataset):
    def __init__(self, train=True):
        self.data = datasets.MNIST(root='./data', train=train, download=True)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)) # Mean and std of MNIST dataset to normalize the images 
        ])

        # Allowed digits and their rotation angles
        self.allowed_digits = [1, 2, 3, 4, 5, 7]
        self.rotation_angles = [0, 90, 180, 270]

        self.samples = []
        for img, label in zip(self.data.data, self.data.targets):
            if label.item() in self.allowed_digits:
                for i, angle in enumerate(self.rotation_angles):
                    rotated_img = TF.rotate(Image.fromarray(img.numpy()), -angle)
                    #Adds rotation and digit labels to each image
                    #Because there is no 0, all digits available mapped to [0,5]
                    self.samples.append((rotated_img, self.allowed_digits.index(label.item()), i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, digit_label, rot_label = self.samples[idx]
        img = self.transform(img)
        return img, digit_label, rot_label


# train and test datasets based on rotated MNIST
train_dataset = RotatedMNIST(train=True)
test_dataset = RotatedMNIST(train=False)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Neural Network for rotated dataset
class RotatedNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential( # 2 Conv layer
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 32x28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 32x14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 64x14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 64x7x7
        )
        self.fc_layers = nn.Sequential( # 1 FC layer
            nn.Linear(64 * 7 * 7, 128), # 128
            nn.ReLU(),
        )
        # Output layers
        self.digit_out = nn.Linear(128, 6)     # 6 classes for digits
        self.rotation_out = nn.Linear(128, 4)  # 4 clasess for rotations

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)

        digit = self.digit_out(x)
        rotation = self.rotation_out(x)

        return digit, rotation

# Train the model cross entropy loss for both digit and rotation classification and Adam optimizer
model = RotatedNN().to(device)
digit_criterion = nn.CrossEntropyLoss()
rotation_criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# i tried 5 epochs but accuracy didnt change significantly after 3 epochs
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    # For counting losses and accuracies
    total_loss = 0
    correct_digit = 0
    correct_rot = 0
    total = 0

    for images, digit_labels, rot_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images, digit_labels, rot_labels = images.to(device), digit_labels.to(device), rot_labels.to(device)

        optimizer.zero_grad()
        digit_pred, rot_pred = model(images)

        digit_loss = digit_criterion(digit_pred, digit_labels)
        rotation_loss = rotation_criterion(rot_pred, rot_labels)
        loss = digit_loss + rotation_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total += images.size(0)
        correct_digit += (digit_pred.argmax(1) == digit_labels).sum().item()
        correct_rot += (rot_pred.argmax(1) == rot_labels).sum().item()

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.2f} | "
          f"Digit Acc: {100 * correct_digit / total:.2f}% | "
          f"Rot Acc: {100 * correct_rot / total:.2f}%")

# Evaluate test dataset
model.eval()
total = 0
correct_digit = 0
correct_rot = 0
test_loss = 0

with torch.no_grad():
    for images, digit_labels, rot_labels in test_loader:
        images, digit_labels, rot_labels = images.to(device), digit_labels.to(device), rot_labels.to(device)
        digit_pred, rot_pred = model(images)
        loss = digit_criterion(digit_pred, digit_labels) + rotation_criterion(rot_pred, rot_labels)
        test_loss += loss.item()
        total += images.size(0)
        correct_digit += (digit_pred.argmax(1) == digit_labels).sum().item()
        correct_rot += (rot_pred.argmax(1) == rot_labels).sum().item()

print(f"\nTest Loss: {test_loss:.4f} | "
      f"Digit Acc: {100 * correct_digit / total:.2f}% | "
      f"Rot Acc: {100 * correct_rot / total:.2f}%")

# Mapping  digit and rotation labels back to their original values because there was no 0 in dataset
digit_labels_map = [1, 2, 3, 4, 5, 7]
rotation_labels_map = [0, 90, 180, 270]

# Testing custom images from user in the current directory 
def predict_images(model, folder_path, device, rows=2, cols=5):
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".png")]
    image_paths = image_paths[:rows * cols]
    # Transformations for the input images to match the training data
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    plt.figure(figsize=(cols * 2, rows * 2.5))
    model.eval()
    # Iterate through the images and make predictions
    with torch.no_grad():
        for i, image_path in enumerate(image_paths):
            image = Image.open(image_path).convert('L')
            input_tensor = transform(image).unsqueeze(0).to(device)

            digit_out, rot_out = model(input_tensor)
            digit_pred = digit_out.argmax(1).item()
            rot_pred = rot_out.argmax(1).item()

            predicted_digit = digit_labels_map[digit_pred]
            predicted_rotation = rotation_labels_map[rot_pred]

            plt.subplot(rows, cols, i + 1)
            plt.imshow(image, cmap='gray')
            plt.title(f"P:{predicted_digit},{predicted_rotation}°", fontsize=9)
            plt.axis('off')

    plt.tight_layout()
    plt.show()

predict_images(model, '.', device, rows=2, cols=5)