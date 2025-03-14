from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('model/plant_desease_model.h5')

# Define a function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        img_array = preprocess_image(file_path)
        prediction = model.predict(img_array)
        os.remove(file_path)
        # Assuming the model returns a class index, you can map it to class names
        class_names = ['Disease A', 'Disease B', 'Healthy']
        predicted_class = class_names[np.argmax(prediction)]
        return render_template('index.html', prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)