#helper functions – helper.py file
import shap
import pickle

# transform categorical data
def transform_categorical(data, d, features_selected):

    for i in list(d.keys()):
        data[i] = d[i].transform(data[i].fillna('NA'))
    return data[features_selected]

# score new data
def score_record(data, clf):

    return clf.predict(data)[0], clf.predict_proba(data)[:,1][0]
