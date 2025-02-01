import torch
import gradio as gr
from PIL import Image
from io import BytesIO
from torchvision import transforms, datasets
from transformers import ResNetForImageClassification

# Paths
MODEL_PATH = "/mnt/c/Users/parth/OneDrive/Documents/ML project/Nike-shoe-classifier/fine_tuned_resnet50.pth"
DATASET_PATH = "/mnt/c/Users/parth/OneDrive/Documents/ML project/Nike-shoe-classifier/Data"

# Load class names
dataset = datasets.ImageFolder(root=DATASET_PATH)
classes = dataset.classes

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
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    model.classifier[1]=torch.nn.Linear(in_features=2048,out_features=2)
    state_dict = torch.load(MODEL_PATH, map_location="cuda", weights_only=False)
    model.load_state_dict(state_dict)
    #model = torch.load(MODEL_PATH, map_location="cuda", weights_only=False)
    model.eval()
    return model

# Initialize model and transform
model = load_model()
transform = get_transform()

# Prediction function
def predict(image):
    image = Image.open(image).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted_idx = torch.max(outputs.logits, 1)
        predicted_class = classes[predicted_idx.item()]
    return predicted_class

# Gradio Interface
demo = gr.Interface(fn=predict, inputs=gr.Image(type="file"), outputs="text", title="Nike Shoe Classifier")

demo.launch()
