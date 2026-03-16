import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
from engine.cnn import Meso4

# --- CONFIGURATION ---
DATASET_PATH = "dataset" # Structure: dataset/real/..., dataset/fake/...
WEIGHTS_SAVE_PATH = "mesonet_weights.pth"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001
IMG_SIZE = 256
# ---------------------

def train_model():
    print(f"Starting training on dataset at: {DATASET_PATH}")
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data Augmentation & Normalization
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load Dataset
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset not found at '{DATASET_PATH}'. Please create folders 'real' and 'fake' inside it.")
        return

    try:
        train_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)
        print(f"Classes found: {train_dataset.classes}")
        
        # Split (simple 80/20 split for demo, or just use all for "full functional project")
        # For simplicity in this script, we train on the whole set provided.
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        print(f"Dataset Size: {len(train_dataset)} images")
        
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Initialize Model
    model = Meso4(num_classes=1).to(device)
    
    # Loss and Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08)

    # Training Loop
    print("\n--- Training Start ---")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.float().to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i+1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] Complete. Avg Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    end_time = time.time()
    print(f"\nTraining Finished in {end_time - start_time:.2f} seconds.")

    # Save Weights
    try:
        torch.save(model.state_dict(), WEIGHTS_SAVE_PATH)
        print(f"SUCCESS: Model weights saved to '{os.path.abspath(WEIGHTS_SAVE_PATH)}'")
        print("You can now restart the application to use the new model.")
    except Exception as e:
        print(f"Error saving weights: {e}")

if __name__ == "__main__":
    train_model()
