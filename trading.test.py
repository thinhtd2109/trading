from flask import Flask, request, jsonify
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load the model and scalers
model = load_model('./predict_multifeature_400.h5')


app = Flask(__name__)

def load_scalers():
    with open('./scalers_400.pkl', 'rb') as f:
        scalers = pickle.load(f)
    return scalers


def preprocess_input(data, scalers):
    """ Preprocess the input JSON data to a format suitable for the model """
    time_step = 60
    features = ['close', 'low', 'high', 'open']
    num_features = len(features)
    input_data = np.zeros((1, time_step, num_features))
    
    # Apply individual scaling and format to input shape
    for i, feature in enumerate(features):
        values = data[feature]
        scaled_values = scalers[feature].transform(np.array(values).reshape(-1, 1)).flatten()
        input_data[0, :, i] = scaled_values
    
    return input_data

@app.route('/predict', methods=['POST'])
def predict_next_price():
    # Expecting JSON input with 60-timestep data for each feature  
    json_data = request.get_json()

    scalers = load_scalers()
    # Preprocess the input to match the model's input shape
    input_data = preprocess_input(json_data, scalers)

    # Predict the next closing price
    next_closing_price = model.predict(input_data)
    print(scalers)
    predicted_price = scalers['close'].inverse_transform(next_closing_price)

    # Return the prediction in JSON format
    return jsonify({'predicted_next_closing_price': float(predicted_price[0, 0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
