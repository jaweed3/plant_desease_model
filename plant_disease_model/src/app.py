from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from utils.preprocess import preprocess_image

app = Flask(__name__)
model = load_model('model/model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    img = preprocess_image(file)
    prediction = model.predict(np.expand_dims(img, axis=0))
    predicted_class = np.argmax(prediction, axis=1)

    return f'Predicted class: {predicted_class[0]}'

if __name__ == '__main__':
    app.run(debug=True)