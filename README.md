# lambda_function.py

import os
import json
import joblib
import pandas as pd
import boto3

# --- Environment Variables ---
# Best practice: Store names of your bucket and model files in environment variables
# In the AWS Lambda configuration, you would set these variables.
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'your-default-bucket-name')
MODEL_FILE_KEY = os.environ.get('MODEL_FILE_KEY', 'housing_model.pkl')
SCALER_FILE_KEY = os.environ.get('SCALER_FILE_KEY', 'scaler.pkl')
COLUMNS_FILE_KEY = os.environ.get('COLUMNS_FILE_KEY', 'model_columns.pkl')

# --- Initialization (Cold Start) ---
# This part of the code runs only when a new Lambda instance is created (a "cold start").
# We load the model here so it's ready in memory for subsequent invocations.
s3_client = boto3.client('s3')
TEMP_MODEL_PATH = '/tmp/model.pkl'
TEMP_SCALER_PATH = '/tmp/scaler.pkl'
TEMP_COLUMNS_PATH = '/tmp/columns.pkl'

try:
    # Download the model artifacts from S3 to the temporary Lambda storage (/tmp)
    s3_client.download_file(S3_BUCKET_NAME, MODEL_FILE_KEY, TEMP_MODEL_PATH)
    s3_client.download_file(S3_BUCKET_NAME, SCALER_FILE_KEY, TEMP_SCALER_PATH)
    s3_client.download_file(S3_BUCKET_NAME, COLUMNS_FILE_KEY, TEMP_COLUMNS_PATH)

    # Load the downloaded files into memory
    model = joblib.load(TEMP_MODEL_PATH)
    scaler = joblib.load(TEMP_SCALER_PATH)
    model_columns = joblib.load(TEMP_COLUMNS_PATH)
    print("Model, scaler, and columns loaded successfully.")

except Exception as e:
    print(f"Error loading model artifacts during initialization: {e}")
    model = None
    scaler = None
    model_columns = None


def lambda_handler(event, context):
    """
    This is the main entry point for the Lambda function.
    'event' contains the data sent from API Gateway.
    """
    print(f"Received event: {event}")

    # Check if the model was loaded correctly during initialization
    if not all([model, scaler, model_columns]):
        return {
            'statusCode': 500,
            'headers': {'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({'error': 'Model is not loaded. Check Lambda logs for initialization errors.'})
        }

    try:
        # The request body from API Gateway is a JSON string in event['body']
        # We need to parse it into a Python dictionary
        body = json.loads(event.get('body', '{}'))
        
        # Convert the incoming JSON data into a pandas DataFrame
        input_df = pd.DataFrame([body])

        # --- Preprocessing (must match training script) ---
        
        # 1. One-hot encode categorical features
        # Reindex ensures the columns match the training data exactly,
        # filling any missing columns with 0.
        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=model_columns, fill_value=0)
        
        # 2. Scale the numerical features using the loaded scaler
        input_scaled = scaler.transform(input_df)
        
        # --- Prediction ---
        # Use the loaded model to make a prediction
        prediction = model.predict(input_scaled)
        
        # The model returns a numpy array, so get the first element
        # Convert numpy float to a standard Python float for JSON serialization
        output = float(prediction[0])

        # Return a successful HTTP response for API Gateway
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*', # Enable CORS for your UI
                'Content-Type': 'application/json'
            },
            'body': json.dumps({'predicted_price': output})
        }

    except Exception as e:
        print(f"Error during prediction: {e}")
        # Return a detailed error response for debugging
        return {
            'statusCode': 400, # Bad Request
            'headers': {'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({'error': f'Failed to process request: {str(e)}'})
        }
