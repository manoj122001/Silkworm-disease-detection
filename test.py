import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Define the file path for the model
filepath = r'E:\silkworm\keras_model.h5'

# Check if the file exists
if os.path.exists(filepath):
    try:
        # Load the model
        model = tf.keras.models.load_model(filepath)
        print("Model Loaded Successfully")
    except Exception as e:
        print("Error loading model:", e)
        exit()  # Exit the program if model loading fails
else:
    print("Model file not found at:", filepath)
    exit()  # Exit the program if model file is not found

# Define a function to predict silk classes
def predict_silk_class(image_path):
    try:
        # Load and preprocess the image, resizing it to match the expected input size
        img = image.load_img(image_path, target_size=(224, 224))  # Resize to (224, 224)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the pixel values

        # Make predictions
        predictions = model.predict(img_array)

        # Get class labels
        class_labels = ["flacherie", "grasserie", "healthy", "muscardine", "pebrine"]

        # Get the predicted class label
        predicted_class = class_labels[np.argmax(predictions)]

        return predicted_class
    except Exception as e:
        print("Error predicting:", e)
        return None

# Example usage:
image_path = r'E:\silkworm\dataset\test\healthy\dt.jpg'
if os.path.exists(image_path):
    predicted_class = predict_silk_class(image_path)
    if predicted_class is not None:
        print(f'Predicted Silk Class: {predicted_class}')
else:
    print("Image file not found at:", image_path)
