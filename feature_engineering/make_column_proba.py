import pickle
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup


def data_to_df(): # read data into df
    df = pd.read_json('../data/data.json')
    return df

def add_fraud(df): # add y-value
    df['fraud'] = df['acct_type'].apply(lambda x: True if x in [u'fraudster_event', u'fraudster', u'fraudster_att'] else False)

def unpack_tix(df): # unpack ticket info
    cost = []
    quantity_sold = []
    quantity_total = []

    for i in range(len(df['ticket_types'])):
        line_cost = []
        line_quant = []
        line_total = []
        for j in range(len(df['ticket_types'].values[i])):
            line_cost.append(df['ticket_types'].values[i][j]['cost'])
            line_quant.append(df['ticket_types'].values[i][j]['quantity_sold'])
            line_total.append(df['ticket_types'].values[i][j]['quantity_total'])
        cost.append(line_cost)
        quantity_sold.append(line_quant)
        quantity_total.append(line_total)

    df['cost'] = cost
    df['quantity_sold'] = quantity_sold
    df['quantity_total'] = quantity_total

def desc_to_nump(df): # unpack descriptions into numpy arrays
    X_des = df['description'].values
    X_org = df['org_desc'].values
    y = df['fraud'].values
    # #less effective
    # for i,row in enumerate(X):
    #     soup = BeautifulSoup(row,'html.parser')
    #     X[i] = soup.get_text()
    return X_des,X_org,y


def unpickle_models(file_p): # unpickle models
    with open(file_p, 'rb') as handle:
        temp = pickle.load(handle)
    return temp

def create_save_file(tf,bnb,file_p,X,y): # create save file for numpy arrays to be saved for export
    #Use np.load('___.npy') to use
    X_tran = tf.transform(X)
    y_proba = bnb.predict_proba(X_tran)[:,1]
    # y_pred = np.empty_like(y)
    # for i,row in enumerate(y_pred):
    #     if i in np.where(y_proba[:,1]>.00001)[0]:
    #         y_pred[i] = 1
    #     else:
    #         y_pred[i] = 0
    np.save(file_p,y_proba)
    pass

def parse_X(X): # html parser
    for i,row in enumerate(X):
        soup = BeautifulSoup(row,'html.parser')
        X[i] = soup.get_text()
    return X


if __name__ == '__main__':
    df = data_to_df()
    unpack_tix(df)
    add_fraud(df)
    X_des,X_org,y = desc_to_nump(df)

    bnb_d = unpickle_models('../pickles/bnb.pickle')
    bnb_od_p = unpickle_models('../pickles/bnb_org_desc.pickle')
    bnb_od_r = unpickle_models('../pickles/bnb_org_desc_rec.pickle')
    tf_d = unpickle_models('../pickles/tf.pickle')
    tf_od_p = unpickle_models('../pickles/tf_org_desc.pickle')
    tf_od_r = unpickle_models('../pickles/tf_org_desc_rec.pickle')

    create_save_file(tf_d,bnb_d,'../data/description_proba.npy',X_des,y) # saving predict_probas
    create_save_file(tf_od_p,bnb_od_p,'../data/org_description_proba_p.npy',X_org,y)
    X_parsed = parse_X(X_org)
    create_save_file(tf_od_r,bnb_od_r,'../data/org_description_proba_r.npy',X_parsed,y)
