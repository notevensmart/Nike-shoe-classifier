import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
from torchvision import transforms, datasets
from transformers.models.resnet import ResNetForImageClassification

# Paths
MODEL_PATH = "/mnt/c/Users/parth/OneDrive/Documents/ML project/Nike-shoe-classifier/fine_tuned_resnet50.pkl"
DATASET_PATH = "/mnt/c/Users/parth/OneDrive/Documents/ML project/Nike-shoe-classifier/Data"

# Load class names
dataset = datasets.ImageFolder(root=DATASET_PATH)
classes = dataset.classes

# FastAPI App
app = FastAPI(title="Nike Shoe Classifier API", version="1.0")

# Transform function
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Load model
def load_model():
    torch.serialization.add_safe_globals([ResNetForImageClassification])
    model = ResNetForImageClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return model

# Initialize model and transform
model = load_model()
transform = get_transform()

# Read image function
def read_image(file: UploadFile):
    image = Image.open(BytesIO(file.file.read())).convert("RGB")
    return transform(image).unsqueeze(0)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image = read_image(file)
        with torch.no_grad():
            outputs = model(image)
            _, predicted_idx = torch.max(outputs.logits, 1)
            predicted_class = classes[predicted_idx.item()]
        return {"Predicted Class": predicted_class}
    except Exception as e:
        return {"Error": str(e)}

# Run API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
