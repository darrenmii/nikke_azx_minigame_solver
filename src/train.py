
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import argparse
from model import DigitNet
from PIL import Image

class CustomDigitDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Explicitly look for folders '0' through '9'
        for i in range(10):
            class_dir = os.path.join(root_dir, str(i))
            if not os.path.exists(class_dir):
                print(f"Warning: Class folder '{i}' not found in {root_dir}")
                continue
                
            files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not files:
                print(f"Warning: Class folder '{i}' is empty")
                continue
                
            for f in files:
                self.samples.append((os.path.join(class_dir, f), i))
                
        if not self.samples:
            raise RuntimeError(f"No valid images found in {root_dir} (checked folders 0-9)")
            
        print(f"Found {len(self.samples)} images across classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('L') # Force grayscale
        if self.transform:
            image = self.transform(image)
        return image, label

def train():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'custom'], help='Dataset to use: mnist or custom')
    parser.add_argument('--data_dir', type=str, default='data/custom_dataset', help='Path to custom dataset')
    args = parser.parse_args()

    # Settings
    batch_size = 64
    epochs = 200 if args.dataset == 'custom' else 5
    lr = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data Transforms
    if args.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        # Custom dataset (likely Grayscale or RGB saved as PNG)
        # We need to ensure it's grayscale 1 channel
        transform = transforms.Compose([
            transforms.RandomAffine(degrees=0, translate=None, scale=(0.5, 1.5), shear=None),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) # Standard mean/std for custom data usually 0.5
        ])

    # Load Data
    if args.dataset == 'mnist':
        print("Downloading MNIST...")
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    else:
        print(f"Loading custom dataset from {args.data_dir}...")
        train_dir = os.path.join(args.data_dir, 'train')
        val_dir = os.path.join(args.data_dir, 'val')
        
        # If no train/val split, just look for root folders 0-9
        if not os.path.exists(train_dir):
            print(f"No train/val split found in {args.data_dir}. Using all data for training (no validation).")
            # Use CustomDigitDataset instead of ImageFolder
            full_dataset = CustomDigitDataset(root_dir=args.data_dir, transform=transform)
            train_dataset = full_dataset
            test_dataset = full_dataset 
        else:
            train_dataset = CustomDigitDataset(root_dir=train_dir, transform=transform)
            test_dataset = CustomDigitDataset(root_dir=val_dir, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

    # Model
    model = DigitNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}")
                running_loss = 0.0

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] Validation Accuracy: {acc:.2f}%")

    # Save Model
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved to model.pth")

if __name__ == "__main__":
    train()
