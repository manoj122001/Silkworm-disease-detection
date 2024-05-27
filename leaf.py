from flask import Flask, render_template, request
import numpy as np
import os

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.image import resize

# Import your model loading code here
from tensorflow.keras.models import load_model

# Define the function to predict the class
def pred_tomato_dieas(model, tomato_plant):
    test_image = load_img(tomato_plant)  # Load image without specifying target_size
    print("@@ Got Image for prediction")
  
    # Resize the image to (224, 224)
    test_image = resize(test_image, (224, 224))

    test_image = img_to_array(test_image) / 255  # Convert image to np array and normalize
    test_image = np.expand_dims(test_image, axis=0)  # Change dimension from 3D to 4D
  
    result = model.predict(test_image)  # Predict disease or not
    print('@@ Raw result = ', result)
  
    pred = np.argmax(result, axis=1)
    print(pred)
    if pred == 0:
        return "flacherie", 'flacherie.html'
    elif pred == 1:
        return "grasserie", 'grasserie.html'
    elif pred == 2:
        return "healthy", 'healthy.html'
    elif pred == 3:
        return "muscardine", 'muscardine.html'
    elif pred == 4:
        return "pebrine", 'pebrine.html'


# Create Flask app instance
app = Flask(__name__)

# Load the model
model = load_model(r'E:\silkworm\keras_model.h5')
print("Model Loaded Successfully")

# Define routes
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']  # Get input file
        filename = file.filename        
        print("@@ Input posted = ", filename)
        
        file_path = os.path.join('E:/silkworm/static/upload', filename)
        file.save(file_path)

        print("@@ Predicting class......")
        pred, output_page = pred_tomato_dieas(model, tomato_plant=file_path)
              
        return render_template(output_page, pred_output=pred, user_image=file_path)

# Run the Flask app
if __name__ == "__main__":
    app.run(threaded=False, port=8080)
