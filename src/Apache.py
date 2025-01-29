from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pickle
import pandas as pd
import os

# Path to the .pkl model
MODEL_PATH = "path/to/model.pkl"

# Sample input data (replace this with your actual input)
SAMPLE_DATA = {"feature1": [5.1], "feature2": [3.5], "feature3": [1.4], "feature4": [0.2]}

# Task 1: Load the model
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

# Task 2: Make predictions
def predict(**context):
    model = context["task_instance"].xcom_pull(task_ids="load_model_task")
    input_data = pd.DataFrame(SAMPLE_DATA)  # Replace with actual input data loading
    predictions = model.predict(input_data)
    print(f"Predictions: {predictions}")
    return predictions

# Task 3: Process results
def process_results(**context):
    predictions = context["task_instance"].xcom_pull(task_ids="predict_task")
    # Example of processing: save predictions to a file
    results_path = "path/to/predictions.csv"
    pd.DataFrame({"predictions": predictions}).to_csv(results_path, index=False)
    print(f"Predictions saved to {results_path}")

# Define the DAG
default_args = {
    "start_date": datetime(2025, 1, 1),
    "retries": 1,
}

with DAG(
    dag_id="model_prediction_pipeline",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:
    load_model_task = PythonOperator(
        task_id="load_model_task",
        python_callable=load_model,
    )

    predict_task = PythonOperator(
        task_id="predict_task",
        python_callable=predict,
        provide_context=True,
    )

    process_results_task = PythonOperator(
        task_id="process_results_task",
        python_callable=process_results,
        provide_context=True,
    )

    # Define task dependencies
    load_model_task >> predict_task >> process_results_task
