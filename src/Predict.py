import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pathlib import Path

# =============================
# Configuration
# =============================
MODEL_PATH = Path("./models/fine_tuned_resnet50_v2.pth")
LABEL_MAP_PATH = Path("./models/label_map.npy")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.85  # Confidence threshold for Real vs Fake

# =============================
# Load Label Map and Model
# =============================
class_names = np.load(LABEL_MAP_PATH, allow_pickle=True)

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# =============================
# Define Transform
# =============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =============================
# Prediction Function
# =============================
def predict_image(image_path, threshold=THRESHOLD):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

        pred_label = class_names[pred_idx.item()]
        confidence = confidence.item()

        # Conservative logic: only label as Real if confidence is high enough
        if pred_label == "Real" and confidence < threshold:
            pred_label = "Fake"

    return pred_label, confidence

# =============================
# Example Usage
# =============================
if __name__ == "__main__":
    test_image = Path("./test_images/example_shoe.jpg")  # Replace with your image path
    label, conf = predict_image(test_image)
    print(f"Prediction: {label} | Confidence: {conf:.3f}")