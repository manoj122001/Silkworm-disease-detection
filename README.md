Silkworm Disease Detection

This repository provides a comprehensive solution for detecting and diagnosing common diseases in silkworms (Bombyx mori) using a deep learning model. The project includes a machine learning model trained on silkworm images and a Flask-based web application for easy and accessible disease prediction.

Table of Contents
Overview
Features
Installation
Usage
Model Training
Project Structure
Contributing
License
Contact
Overview
The Silkworm Disease Detection project is designed to help researchers, farmers, and enthusiasts quickly identify and diagnose common silkworm diseases. The project leverages a convolutional neural network (CNN) to classify diseases based on silkworm images.

Features
Disease Classification: Identifies common silkworm diseases such as Flacherie, Grasserie, Muscardine, and Pebrine.
User-Friendly Web Interface: Upload images and view results easily.
Machine Learning: Employs a deep learning model for accurate disease prediction.
Robust and Scalable: Easily extendable with more data or additional features.
Installation
Prerequisites
Python 3.7+
Git
Virtual Environment (recommended)
TensorFlow and Keras
Clone the Repository
bash
Copy code
git clone https://github.com/yourusername/silkworm-disease-detection.git
cd silkworm-disease-detection
Create a Virtual Environment (Optional but Recommended)
bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install Required Dependencies
bash
Copy code
pip install -r requirements.txt
Model File
Make sure to place the trained model file my_model.h5 in the root directory or update the path in the code accordingly. You can also train the model using the provided training script (details below).

Additional Setup
Ensure you have a directory static/upload for saving uploaded images.
Make sure all necessary HTML templates are available in the templates directory.
Usage
Running the Web Application
Start the Flask server:

bash
Copy code
python app.py
Open your browser and navigate to:

arduino
Copy code
http://localhost:8080
Upload an image of a silkworm and click the "Predict" button to diagnose the disease.

Example Input and Output
Upload Image: Select an image of a silkworm.
View Prediction: The application displays the predicted disease and the uploaded image.

Model Training
Training Steps
Preprocessing and Augmentation:
Applies various transformations such as rotation, shifting, shearing, and zooming to increase the diversity of the training set.
Model Architecture:
Uses convolutional layers followed by batch normalization and pooling layers to extract features.
Includes dropout to prevent overfitting.
Training Process:
Incorporates a learning rate scheduler to adjust the learning rate during training.
Uses class weights to handle imbalanced datasets.
Project Structure
php
Copy code
silkworm-disease-detection/
│
├── static/
│   ├── upload/          # Directory for uploaded images
│   └── assets/          # Additional assets like images
│
├── templates/           # HTML files for rendering web pages
│   ├── index.html
│   ├── flacherie.html
│   ├── grasserie.html
│   ├── healthy.html
│   ├── muscardine.html
│   ├── pebrine.html
│   └── error.html
│
├── app.py               # Main Flask application code
├── training.py          # Model training script
├── requirements.txt     # Python dependencies
└── README.md            # This README file
Contributing
We welcome contributions from the community! Here's how you can contribute:

Fork the repository.
Create a new branch for your feature or bug fix.
Commit your changes and push them to your branch.
Submit a pull request.
Please follow the coding standards and guidelines provided in CONTRIBUTING.md.

Coding Standards
Adhere to PEP 8 guidelines for Python code.
Ensure your code is well-documented and includes meaningful comments.
Write tests for any new features or bug fixes.
