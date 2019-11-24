# Serve model as a flask application
# Code mostly from https://github.com/tanujjain/deploy-ml-model
import numpy as np
from flask import Flask, request
from models import get_model_classif_nasnet
from utils import read_image, preprocess_input, read_image_base64
from io import BytesIO

model = None
app = Flask(__name__)


def load_model(h5_path):
    global model
    # model variable refers to the global variable
    model = get_model_classif_nasnet()
    model.load_weights(h5_path)


@app.route('/')
def home_endpoint():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def get_prediction():
    # Works only for a single sample
    if request.method == 'POST':
        data = request.data
        data = np.array(preprocess_input(read_image(BytesIO(data))))[np.newaxis, ...]
        prediction = model.predict(data).ravel().tolist()
    return str(prediction[0])


if __name__ == '__main__':
    h5_path = "model.h5"
    load_model(h5_path)
    app.run(host='0.0.0.0', port=80)
