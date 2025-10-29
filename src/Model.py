import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm

# =============================
# Configuration
# =============================
SEED = 42
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
DATA_DIR = Path("./Data")  # Relative path for portability
MODEL_SAVE_PATH = Path("./models/fine_tuned_resnet50_v2.pth")
LABEL_MAP_PATH = Path("./models/label_map.npy")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# Reproducibility
# =============================
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =============================
# Data Augmentation
# =============================
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =============================
# Dataset and DataLoader
# =============================
dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)
class_names = dataset.classes
np.save(LABEL_MAP_PATH, class_names)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =============================
# Model Setup
# =============================
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last block for fine-tuning
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace classifier head
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))
model = model.to(DEVICE)

# =============================
# Training Components
# =============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# =============================
# Training Loop
# =============================
def train_one_epoch(epoch):
    model.train()
    total_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss

# =============================
# Validation Loop
# =============================
def validate():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# =============================
# Main Training Routine
# =============================
if __name__ == "__main__":
    print(f"Training on {DEVICE} with {len(class_names)} classes: {class_names}")
    best_acc = 0.0

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(epoch)
        val_acc = validate()
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ Model saved (Val Acc: {best_acc:.2f}%)")

    print(f"Training complete. Best Validation Accuracy: {best_acc:.2f}%")
