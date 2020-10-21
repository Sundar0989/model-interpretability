from flask import Flask, request, redirect, url_for, flash, jsonify, make_response
import pandas as pd
import numpy as np
import pickle
import json
import shap
from helper import *

# read pickle files
with open('score_objects.pkl', 'rb') as handle:
    d, features_selected, clf, explainer = pickle.load(handle)

app = Flask(__name__)

@app.route('/api/', methods=['POST'])
def makecalc():

    json_data = request.get_json()
    #read the real time input to pandas df
    data = pd.DataFrame(json_data)
    #transform DataFrame
    data = transform_categorical(data, d, features_selected)
    #score df
    prediction, probability = score_record(data, clf)
    #convert predictions to dictionary
    data['prediction'] = prediction
    data['probability'] = probability
    output = data.to_dict(orient='rows')[0]
    #output['plot'] = p
    return jsonify(output)

if __name__ == '__main__':

    app.run(debug=True, host='0.0.0.0', port=5000)
