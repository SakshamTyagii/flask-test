import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.linear_model import LogisticRegression
import joblib

app = Flask(__name__)
CORS(app)

loaded_data = joblib.load('model.pkl')

# Extract the model and column names
model = loaded_data[0]  # The first element is the model
columns = loaded_data[1] 

@app.route('/predict', methods=['POST'])
def predict():
    # Get the user input from the request body
    data = request.get_json()
    print(data)
    nitrogen = data['nitrogen']
    phosphorus = data['phosphorus']
    potassium = data['potassium']
    ph = data['ph']
    temperature = data['temperature']
    humidity = data['humidity']
    rainfall = data['rainfall']

    # Create a pandas dataframe from the input data
    input_dict = {
    'nitrogen': nitrogen,
    'phosphorus': phosphorus,
    'potassium': potassium,
    'temperature': temperature,
    'humidity': humidity,
    'ph': ph,
    'rainfall': rainfall
}

# Create a Pandas DataFrame from the dictionary
    input_data = pd.DataFrame([input_dict])

    print(type(model))
    print(input_data)
    # Make predictions using your model
    y_pred = model.predict(input_data)
    print(y_pred)
    # Return the predicted result as JSON
    return jsonify({'prediction': y_pred[0]})

if __name__ == '__main__':
    app.run(debug=True)