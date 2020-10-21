import streamlit as st
import requests
import datetime
import shap
import json
import pickle
import pandas as pd
import streamlit.components.v1 as components
import matplotlib.pyplot as plt


st.title('Python Real Time Scoring API + Model Explainer')

# PySpark API endpoint
url = 'http://pythonapi:5000'
endpoint = '/api/'

# description and instructions
st.write('''A real time scoring API for Python model.''')

st.sidebar.header('User Input features')

def user_input_features():
    input_features = {}
    input_features["age"] = st.sidebar.slider('Age', 18, 95)
    input_features["education"] = st.sidebar.selectbox('Education Qualification', ['tertiary', 'secondary', 'unknown', 'primary'])
    input_features["balance"]= st.sidebar.slider('Current balance', -10000, 150000)
    input_features["housing"] = st.sidebar.selectbox('Do you own a home?', ['yes', 'no'])
    input_features["loan"] = st.sidebar.selectbox('Do you have a loan?', ['yes', 'no'])
    input_features["contact"] = st.sidebar.selectbox('Best way to contact you', ['cellular', 'telephone', 'unknown'])
    date = st.sidebar.date_input("Today's Date")
    input_features["day"] = date.day
    input_features["month"] = date.strftime("%b").lower()
    input_features["duration"] = st.sidebar.slider('Duration', 0, 5000)
    input_features["campaign"] = st.sidebar.slider('Campaign', 1, 63)
    input_features["pdays"] = st.sidebar.slider('pdays', -1, 871)
    input_features["previous"] = st.sidebar.slider('previous', 0, 275)
    input_features["poutcome"] = st.sidebar.selectbox('poutcome', ['success', 'failure', 'other', 'unknown'])
    return [input_features]

json_data = user_input_features()

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# read pickle files
with open('score_objects.pkl', 'rb') as handle:
    d, features_selected, clf, explainer = pickle.load(handle)

# explain model prediction results
def explain_model_prediction(data):
    # Calculate Shap values
    shap_values = explainer.shap_values(data)
    p = shap.force_plot(explainer.expected_value[1], shap_values[1], data)
    return p, shap_values

submit = st.sidebar.button('Get predictions')
if submit:
    results = requests.post(url+endpoint, json=json_data)
    results = json.loads(results.text)
    results = pd.DataFrame([results])

    st.header('Final Result')
    prediction = results["prediction"]
    probability = results["probability"]

    st.write("Prediction: ", int(prediction))
    st.write("Probability: ", round(float(probability),3))

    #explainer force_plot
    results.drop(['prediction', 'probability'], axis=1, inplace=True)
    results = results[features_selected]
    p, shap_values = explain_model_prediction(results)
    st.subheader('Model Prediction Interpretation Plot')
    st_shap(p)

    st.subheader('Summary Plot 1')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.summary_plot(shap_values[1], results)
    st.pyplot(fig)

    st.subheader('Summary Plot 2')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.summary_plot(shap_values[1], results, plot_type='bar')
    st.pyplot(fig)
