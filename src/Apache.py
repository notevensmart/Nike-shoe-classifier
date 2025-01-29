from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pickle
import os
from PIL import Image
import torchvision.transforms as transforms
import torch

# Path to the .pkl model
MODEL_PATH = "C:/Users/parth/OneDrive/Documents/ML project/fine_tuned_resnet50.pkl"
# Path to the input image
IMAGE_PATH = "C:/Users/parth/OneDrive/Documents/ML project/Screenshot 2025-01-21 185137.png"

# Define a transform for the image (customize based on your model)
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the expected model input size
    transforms.ToTensor(),         # Convert to Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ImageNet normalization
])

# Task 1: Load the model
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully")
    return model

# Task 2: Preprocess image
def preprocess_image():
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Image file not found at {IMAGE_PATH}")
    image = Image.open(IMAGE_PATH).convert("RGB")  # Ensure it's an RGB image
    transformed_image = image_transform(image)
    print("Image preprocessed successfully")
    return transformed_image.unsqueeze(0)  # Add batch dimension

# Task 3: Make predictions
def predict(**context):
    model = context["task_instance"].xcom_pull(task_ids="load_model_task")
    image = context["task_instance"].xcom_pull(task_ids="preprocess_image_task")

    if model is None:
        raise ValueError("Model could not be retrieved from previous task")
    if image is None:
        raise ValueError("Image could not be retrieved from previous task")
    
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        predictions = model(image)
    
    predicted_class = predictions.argmax(dim=1).item()
    print(f"Predicted class: {predicted_class}")
    return predicted_class

# Task 4: Process results
def process_results(**context):
    predicted_class = context["task_instance"].xcom_pull(task_ids="predict_task")
    if predicted_class is None:
        raise ValueError("Prediction could not be retrieved from previous task")
    
    # Save predictions or take action
    results_path = "path/to/predictions.txt"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)  # Ensure directory exists
    with open(results_path, "w") as f:
        f.write(f"Predicted class: {predicted_class}\n")
    print(f"Predictions saved to {results_path}")

# Define the DAG
default_args = {
    "start_date": datetime(2025, 1, 1),
    "retries": 1,
}

with DAG(
    dag_id="image_classification_pipeline",
    default_args=default_args,
    schedule=None,  # Using "schedule" instead of the deprecated "schedule_interval"
    catchup=False,
) as dag:
    load_model_task = PythonOperator(
        task_id="load_model_task",
        python_callable=load_model,
    )

    preprocess_image_task = PythonOperator(
        task_id="preprocess_image_task",
        python_callable=preprocess_image,
    )

    predict_task = PythonOperator(
        task_id="predict_task",
        python_callable=predict,
    )

    process_results_task = PythonOperator(
        task_id="process_results_task",
        python_callable=process_results,
    )

    # Define task dependencies
    load_model_task >> preprocess_image_task >> predict_task >> process_results_task
