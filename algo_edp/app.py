from flask import Flask, request, jsonify
import pickle
import numpy as np
from PIL import Image
from edp_changed import analysis
import cv2
import io
from keras.preprocessing.image import img_to_array
app = Flask(__name__)

# # Load your pre-trained model
# with open('model.pkl', 'rb') as f:
#     model = pickle.load(f)

@app.route('/', methods=['GET'])
def index():
    return 'Server is running'


@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file is sent in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image']

    # Convert the image to numpy array
    frame = Image.open(image.stream)

    frame_array = np.array(frame)
    image_data=analysis(frame_array)

    # Return the prediction as JSON response
    return jsonify({'prediction': image_data.tolist()}), 200


if __name__ == '__main__':
    app.run(debug=True)