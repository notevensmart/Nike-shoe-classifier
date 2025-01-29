import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import ResNetForImageClassification

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the saved model
model_path = "fine_tuned_resnet50.pkl"  # Adjust path if needed
model = torch.load(model_path, map_location=device)
model = model.to(device)
model.eval()

# Define transformations (must match what was used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define inference function
def predict_image(image_path, model, transform, class_names):
    """Predicts the class of a given image."""
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image)
        logits = outputs.logits  # Extract logits
        probabilities = F.softmax(logits, dim=1)  # Convert to probabilities
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        predicted_class = class_names[predicted_class_idx]
        confidence = probabilities[0, predicted_class_idx].item()  
        return predicted_class, confidence

# Example Usage
if __name__ == "__main__":
    image_path = "C:/Users/parth/OneDrive/Documents/ML project/Screenshot 2025-01-21 185137.png"  # Replace with actual test image
    class_names = ["Fake", "Real"]  # Modify based on training classes

    predicted_class, probability = predict_image(image_path, model, transform, class_names)
    print(f"Predicted class: {predicted_class} with confidence: {probability:.4f}")