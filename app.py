"""
Bengaluru House Price Prediction Flask Application

This application serves a machine learning model that predicts house prices in Bengaluru
based on location, square footage, number of bathrooms, and BHK configuration.

Features:
- Home page with input form
- Price prediction endpoint
- Error handling
- Model loading at startup
"""

import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
import json

app = Flask(__name__)

# Load the trained model and columns data
MODEL_PATH = 'banglore_home_prices_model.pickle'
COLUMNS_PATH = 'columns.json'

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    with open(COLUMNS_PATH, 'r') as f:
        columns_data = json.load(f)
        data_columns = columns_data['data_columns']
except Exception as e:
    print(f"Error loading model or columns data: {str(e)}")
    model = None
    data_columns = []

@app.route('/')
def home():
    """Render the home page with input form"""
    locations = sorted([col for col in data_columns if col not in ['total_sqft', 'bath', 'bhk']])
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests and return results"""
    if request.method == 'POST':
        try:
            # Get form data
            location = request.form['location'].lower()
            total_sqft = float(request.form['total_sqft'])
            bath = int(request.form['bath'])
            bhk = int(request.form['bhk'])
            
            # Validate inputs
            if total_sqft <= 0 or bath <= 0 or bhk <= 0:
                return render_template('index.html', 
                                    error="All values must be positive numbers",
                                    locations=sorted([col for col in data_columns if col not in ['total_sqft', 'bath', 'bhk']]))
            
            # Prepare input for prediction
            loc_index = np.where(np.array(data_columns) == location)[0]
            
            if len(loc_index) == 0:
                return render_template('index.html', 
                                    error="Selected location is not available in our database",
                                    locations=sorted([col for col in data_columns if col not in ['total_sqft', 'bath', 'bhk']]))
            
            x = np.zeros(len(data_columns))
            x[0] = total_sqft
            x[1] = bath
            x[2] = bhk
            if loc_index >= 0:
                x[loc_index] = 1
                
            # Make prediction
            price = round(model.predict([x])[0], 2)
            
            # Format price with commas
            formatted_price = "{:,.2f}".format(price)
            
            return render_template('result.html', 
                                 prediction_text=f'Estimated Price: ₹{formatted_price}',
                                 location=location.capitalize(),
                                 total_sqft=total_sqft,
                                 bath=bath,
                                 bhk=bhk)
            
        except Exception as e:
            return render_template('index.html', 
                                error=f"An error occurred: {str(e)}",
                                locations=sorted([col for col in data_columns if col not in ['total_sqft', 'bath', 'bhk']]))
    
    return render_template('index.html', locations=sorted([col for col in data_columns if col not in ['total_sqft', 'bath', 'bhk']]))

@app.route('/get_locations')
def get_locations():
    """API endpoint to get available locations"""
    locations = sorted([col for col in data_columns if col not in ['total_sqft', 'bath', 'bhk']])
    return jsonify({'locations': locations})

if __name__ == "__main__":
    app.run(debug=True)