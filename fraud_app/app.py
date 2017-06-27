import sqlite3
import datetime
# from bokeh.plotting import figure
# from bokeh.embed import components
# from bokeh.resources import INLINE
from flask import Flask, render_template
import predict
import pickle
import pandas as pd
import numpy as np
import requests
import json

# convert_data_and_return_pred(df,bnb_d,bnb_od_p,bnb_od_r,tf_d,tf_od_p,tf_od_r,rf_mod)

# from data import get_rates, get_lowest_rate, add_data

def unpickle_models(file_p):
    with open(file_p, 'rb') as handle:
        temp = pickle.load(handle)
    return temp

bnb_d = unpickle_models('pickles/bnb.pickle')
bnb_od_p = unpickle_models('pickles/bnb_org_desc.pickle')
bnb_od_r = unpickle_models('pickles/bnb_org_desc_rec.pickle')
tf_d = unpickle_models('pickles/tf.pickle')
tf_od_p = unpickle_models('pickles/tf_org_desc.pickle')
tf_od_r = unpickle_models('pickles/tf_org_desc_rec.pickle')
rf_mod = unpickle_models('pickles/randomforest_fraud.pickle')

app = Flask(__name__)

# We'd normally include configuration settings in this call

@app.route('/')
def index():
    new_data = requests.get('http://galvanize-case-study-on-fraud.herokuapp.com/data_point').text
    nd_j = json.loads(new_data)
    df = pd.DataFrame.from_dict(nd_j, orient='index').T
    name = df['name']
    df2 = predict.main()

    if df2['fraud'][0] == 1:
        is_fraud = "Fraud"
    else:
        is_fraud = "Not Fraud"
    if df2['priority'][0]== 0:
        prio = ":)"
    elif df2['priority'][0]== 1:
        prio = "Low Priority"
    elif df2['priority'][0]== 2:
        prio = "Medium Priority"
    else:
        prio = "High Priority"

    return render_template(
        'index.html',
        fraud = is_fraud,
        priority = prio,
        name = name)

@app.route('/index')
def index2():
    new_data = requests.get('http://galvanize-case-study-on-fraud.herokuapp.com/data_point').text
    nd_j = json.loads(new_data)
    df = pd.DataFrame.from_dict(nd_j, orient='index').T
    name = df['name']
    df2 = convert_data_and_return_pred(df,bnb_d,bnb_od_p,bnb_od_r,tf_d,tf_od_p,tf_od_r,rf_mod)

    if df2['fraud'][0] == 1:
        is_fraud = "FRAUD"
    else:
        is_fraud = "not fraud"
    if df2['priority'][0]== 0:
        prio = ""
    elif df2['priority'][0]== 1:
        prio = "low"
    elif df2['priority'][0]== 2:
        prio = "med"
    else:
        prio = "HIGH"

    return render_template(
        'index.html',
        fraud = is_fraud,
        priority = prio,
        name = name)


if __name__ == '__main__':

    app.run(host='0.0.0.0', port=8105, debug=True)
