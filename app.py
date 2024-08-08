from flask import Flask, render_template, request, redirect, url_for, session, send_file
import os
from ultralytics import YOLO
import numpy as np
import io
import sys
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)

# Set a secret key for session management if needed
app.secret_key = os.urandom(24)  # Automatically generates a random secret key

# Function to get the latest model path dynamically
def get_latest_model_path(directory='runs/classify'):
    subdirs = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    latest_subdir = max(subdirs, key=os.path.getmtime)
    model_path = os.path.join(latest_subdir, 'weights', 'last.pt')
    return model_path

# Function to get the latest results CSV path dynamically
def get_latest_results_path(directory='runs/classify'):
    subdirs = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    latest_subdir = max(subdirs, key=os.path.getmtime)
    results_path = os.path.join(latest_subdir, 'results.csv')
    return results_path

# Function to generate plots from results
def generate_training_plots(results_path):
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")

    print(f"Loading results from: {results_path}")  # Debug statement

    results = pd.read_csv(results_path)

    # Print first few rows to debug
    print("Results DataFrame head:")
    print(results.head())

    # Strip whitespace from column names
    results.columns = results.columns.str.strip()

    # Check for required columns
    required_columns = ['epoch', 'train/loss', 'metrics/accuracy_top1', 'val/loss']
    missing_columns = [col for col in required_columns if col not in results.columns]
    if missing_columns:
        raise KeyError(f"Required columns are missing from the results CSV: {', '.join(missing_columns)}")

    # Ensure static directory exists
    if not os.path.exists('static'):
        os.makedirs('static')

    # Loss vs Epochs
    try:
        plt.figure()
        plt.plot(results['epoch'], results['train/loss'], label='Train Loss')
        plt.plot(results['epoch'], results['val/loss'], label='Validation Loss', color='red')
        plt.grid()
        plt.title('Loss vs Epochs')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        plt.savefig('static/loss_vs_epochs.png')
        plt.close()
        print("Loss vs Epochs plot saved successfully.")
    except Exception as e:
        print(f"Error saving Loss vs Epochs plot: {e}")

    # Validation Accuracy vs Epochs
    try:
        plt.figure()
        plt.plot(results['epoch'], results['metrics/accuracy_top1'] * 100)
        plt.grid()
        plt.title('Validation Accuracy vs Epochs')
        plt.ylabel('Accuracy (%)')
        plt.xlabel('Epochs')
        plt.savefig('static/accuracy_vs_epochs.png')
        plt.close()
        print("Validation Accuracy vs Epochs plot saved successfully.")
    except Exception as e:
        print(f"Error saving Validation Accuracy vs Epochs plot: {e}")

# Home route to display options
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(url_for('index'))

        file = request.files['file']
        if file.filename == '':
            return redirect(url_for('index'))

        # Save the uploaded image to the static directory
        image_path = os.path.join('static', file.filename)
        file.save(image_path)

        # Load the latest model
        model_path = get_latest_model_path()
        model = YOLO(model_path)

        # Predict on the uploaded image
        results = model(image_path)
        names_dict = results[0].names
        probs = results[0].probs.data.tolist()

        predicted_class = names_dict[np.argmax(probs)]
        accuracy = max(probs) * 100

        return render_template('result.html', predicted_class=predicted_class, accuracy=accuracy, image_path=file.filename)

    return render_template('predict.html')

# Route to handle training
@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        epochs = int(request.form['epochs'])
        model_path = "yolov8n-cls.pt"
        model = YOLO(model_path)

        # Capture training output
        output = io.StringIO()
        sys.stdout = output
        model.train(data='dataset', epochs=epochs, imgsz=64)
        sys.stdout = sys.__stdout__

        log = output.getvalue()
        session['log'] = log  # Store log in session

        # Generate and save training plots
        try:
            results_path = get_latest_results_path()
            generate_training_plots(results_path)
        except (FileNotFoundError, KeyError) as e:
            return render_template('train.html', message=f"Training completed but error generating plots: {e}", log=log)

        return render_template('train.html', message="Training completed successfully!", log=log)
    return render_template('train.html')

# Route to show training options
@app.route('/train_options')
def train_options():
    return render_template('train_options.html')

# Route to serve training plots
@app.route('/training_plots')
def training_plots():
    return render_template('training_plots.html')

# Route to serve plot images
@app.route('/plot/<filename>')
def plot(filename):
    return send_file(os.path.join('static', filename))

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
