from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from transformers import pipeline
from transformers import AutoImageProcessor, ResNetForImageClassification
import tensorflow as tf
import kagglehub
import zipfile
import os
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader , random_split
from torch import optim
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F


custom_path = "C:/Users/parth/OneDrive/Documents/ML project/Nike-shoe-classifier/Data"
#os.makedirs(custom_path, exist_ok=True)
#with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    #zip_ref.extractall(custom_path)

##path = kagglehub.dataset_download("eliasdabbas/nike-shoes-images", path=custom_path)
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Device Count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current Device Index: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is not available. Check installation or drivers.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = torch.load("fine_tuned_resnet50.pkl", map_location=device)

# Move to GPU if available
model = model.to(device)

# Set to evaluation mode
model.eval()


# Step 2: Data preprocessing and loading
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard for pre-trained models
])


# Apply the transformations

dataset = datasets.ImageFolder(root=custom_path, transform=transform) 
# Step 3: Train-Validation Split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

num_classes = len(dataset.classes)
# DataLoaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Step 4: Load Pre-trained ResNet and Modify for Custom Classes
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, num_classes) 
model = model.to(device)

# Step 5: Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Step 6: Training Loop
epochs = 5
for epoch in range(epochs):
    model.train()
    train_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)  # Output is an object, not raw logits
        logits = outputs.logits  # Extract logits
        loss = criterion(logits, labels)  # Compute loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}")

# Step 7: Validation Loop
model.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        logits = outputs.logits  # Extract logits
        _, preds = torch.max(logits, 1)  # Predicted class
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())


# Step 8: Compute Validation Accuracy
accuracy = (np.array(all_labels) == np.array(all_preds)).mean()
print(f"Validation Accuracy: {accuracy:.2f}")

# Step 9: Save the Fine-Tuned Model
torch.save(model, "fine_tuned_resnet50.pkl")
model = model.to(device)
print("Model saved as 'fine_tuned_resnet50.pkl'")

# Step 10: Inference on New Images
def predict_image(image_path, model, transform, classes):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        logits = outputs.logits
        _, predicted_class = torch.max(logits, 1)
        probs = F.softmax(logits, dim=1)  # Apply softmax to get probabilities
        
        predicted_class = torch.argmax(probs, dim=1).item()  # Get the class index
        confidence = probs[0, predicted_class].item()
    return classes[predicted_class.item()] , confidence
image_path = "C:/Users/parth/OneDrive/Documents/ML project/Screenshot 2025-01-21 185137.png"
predicted_class , confidence= predict_image(image_path, model, transform, dataset.classes)
print(f"The predicted class for the image is: {predicted_class} , the probabilty is : {confidence: .2f}")

