# Classification of Electronic Components Using YOLOv8 model (Python with AIML)

## Project Overview

This project aims to classify electronic components using the YOLOv8 (You Only Look Once version 8) model, leveraging Python and artificial intelligence/machine learning (AIML) techniques. The application provides functionalities to train the model on custom datasets, make predictions on images, and visualize training results.

## Features

- **Model Training**: Train the YOLOv8 model on a custom dataset with user-specified epoch count.
- **Prediction**: Upload an image for prediction using the trained model.
- **Visualization**: View training loss and validation accuracy plots.
- **Dynamic File Handling**: Automatically use the most recent model and results files.

## Installation

To set up the project, follow these steps:

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/rishiganesht2003/Classification-of-Electronic-Components-Using-YOLOv8-model-Python-with-AIML-.git
    cd Classification-of-Electronic-Components-Using-YOLOv8-model-Python-with-AIML
    ```

2. **Create and Activate a Virtual Environment**:

    ```bash
    python -m venv venv
    source venv/bin/activate  

#### On Windows use 
    
    venv\Scripts\activate


3. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Start the Flask Application**:

    ```bash
    python app.py
    ```

2. **Access the Application**:

    Open a web browser and navigate to `http://127.0.0.1:5000`.

3. **Train the Model**:

    - Go to the "Train Model" page.
    - Upload your dataset and specify the number of epochs.
    - The application will train the model and generate plots.

4. **Predict with the Model**:

    - Go to the "Predict Image" page.
    - Upload an image for classification.
    - View the predicted class and confidence.

5. **View Training Plots**:

    - Access the "Training Plots" page to view generated plots for training loss and validation accuracy.

## Files and Directories

- **`.idea/`**: Directory used by the JetBrains IDE for project-specific settings.
- **`dataset/`**: Directory containing training and validation datasets.(Increase the number of images of each class to increare accuracy in prediction)
- **`runs/`**: Directory where training results and model weights are stored.
- **`static/`**: Directory for storing static files like CSS and plot images.
- **`templates/`**: Directory for HTML templates.
- **`app.py`**: Main Flask application file.
- **`requirements.txt`**: File listing Python package dependencies.
- **`yolov8n-cls.pt`**: Pre-trained YOLOv8 model file.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [YOLOv8](https://github.com/ultralytics/yolov5) for object detection.
- [Flask](https://flask.palletsprojects.com/) for web application framework.
- [Pandas](https://pandas.pydata.org/) for data manipulation.
- [Matplotlib](https://matplotlib.org/) for plotting.

