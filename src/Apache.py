import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
from torchvision import transforms, models, datasets
import torch.nn as nn

# Paths
MODEL_PATH = "/mnt/c/Users/parth/OneDrive/Documents/ML project/Nike-shoe-classifier/fine_tuned_resnet50.pkl"
DATASET_PATH = "/mnt/c/Users/parth/OneDrive/Documents/ML project/Nike-shoe-classifier/Data"

# Load class names from dataset
dataset = datasets.ImageFolder(root=DATASET_PATH)
classes = dataset.classes  # List of class names

# Initialize FastAPI
app = FastAPI(title="Nike Shoe Classifier API", version="1.0")

# Define image transformation function
def get_transform():
    """Returns the transformation pipeline for input images."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Load the model
def load_model():
    """Loads the fine-tuned ResNet50 model without external dependencies."""
    num_classes = len(classes)
    
    # Load a pre-trained ResNet-50 model
    model = models.resnet50(pretrained=False)
    
    # Modify the fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # Load saved model weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

# Initialize model and transform
model = load_model()
transform = get_transform()

# Helper function to preprocess the image
def read_image(file: UploadFile):
    """Reads and preprocesses an image file for model inference."""
    image = Image.open(BytesIO(file.file.read())).convert("RGB")
    image = transform(image).unsqueeze(0)  # Apply transformations and add batch dimension
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    API endpoint that receives an image file, processes it,
    and returns the predicted class.
    """
    try:
        image = read_image(file)
        with torch.no_grad():
            outputs = model(image)
            _, predicted_idx = torch.max(outputs, 1)
            predicted_class = classes[predicted_idx.item()]

        return {"Predicted Class": predicted_class}

    except Exception as e:
        return {"Error": str(e)}

# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
