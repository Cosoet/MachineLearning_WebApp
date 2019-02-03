#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, jsonify, render_template

# Helper function for predictions
def predict(f1, f2, f3, f4):
    prediction = {'Prediction': int(0)}

    x_input = np.array([f1, f2, f3, f4]).reshape(1,4)

    model_file_path = 'iris_rfc.pkl'
    rfc = pickle.load(open(model_file_path, 'rb'))
    prediction['Prediction'] = int(rfc.predict(x_input)[0])
    
    print (prediction)
    return prediction

# Initiate a server.
app = Flask(__name__)

# Web server application
@app.route("/")
def index():
    return "IRIS Prediction!"

@app.route("/iris_prediction", methods = ['POST', 'GET'])
def iris_predict():
    if request.method == 'POST':
        result = predict(f1=request.form['p1'], f2=request.form['p2'], f3=request.form['p3'], f4=request.form['p4'])
        return render_template('iris.html', result = result)
    else:
        return render_template('iris.html')

if __name__ == "__main__":
    app.run(debug=True)





