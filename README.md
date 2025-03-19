# Plant Disease Classification Model

This project is a Flask web application that allows users to upload images of plants and receive predictions on potential diseases using a Convolutional Neural Network (CNN) implementation with Tensorflow model.

This Projects is using Kaggle Plant Desease Dataset. And improved with the Tensorflow preprocessing reached 90% accuracy.

You can get the Dataset here `https://www.kaggle.com/datasets/emmarex/plantdisease`

## Project Structure

```
plant_disease_model
├── src
│   ├── app.py                # Entry point of the Flask application
│   ├── model
│   │   └── model.h5         # Trained TensorFlow model for plant disease classification
│   ├── static
│   │   └── styles.css        # CSS styles for the web application
│   ├── templates
│   │   └── index.html        # Main HTML template for the web application
│   └── utils
│       └── preprocess.py     # Utility functions for preprocessing input images
├── requirements.txt          # Project dependencies
└── README.md                 # Documentation for the project
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd plant_disease_model
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Flask application:
   ```
   python src/app.py
   ```

2. Open your web browser and go to `http://127.0.0.1:5000`.

3. Use the provided form to upload an image of a plant leaf. The application will process the image and display the predicted disease.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
