import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import precision_score, recall_score
import pickle
from bs4 import BeautifulSoup

# read data into df
def data_to_df():
    df = pd.read_json('data/data.json')
    return df

# add y-value column
def add_fraud(df):
    df['fraud'] = df['acct_type'].apply(lambda x: True if x in [u'fraudster_event', u'fraudster', u'fraudster_att'] else False)
    # df.drop('acct_type', axis=1, inplace=True)

# unpack ticket info
def unpack_tix(df):
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

# unpack descriptions into numpy arrays
def desc_to_nump(df):
    X = df['org_desc'].values
    y = df['fraud'].values
    #less effective
    for i,row in enumerate(X):
        soup = BeautifulSoup(row,'html.parser')
        X[i] = soup.get_text()
    return X,y

# create TF-IDF Vectorizer
def build_model(X_train,y_train):
    tf = TfidfVectorizer() #stop_words='english'
    bnb = BernoulliNB()
    X_tran = tf.fit_transform(X_train)
    bnb.fit(X_tran,y_train)
    return tf,bnb

# score model on test sample
def score_train_test(tf,bnb, X_train, X_test, y_train, y_test):
    X_tran_tr = tf.transform(X_train)
    y_proba = bnb.predict_proba(X_tran_tr)
    y_pred_tr = np.empty_like(y_train)
    for i,row in enumerate(y_pred_tr):
        if i in np.where(y_proba[:,1]>.00001)[0]:
            y_pred_tr[i] = True
        else:
            y_pred_tr[i] = False
    print("train recall:  {}".format(recall_score(y_train,y_pred_tr)))
    print("train precision:  {}".format(precision_score(y_train,y_pred_tr)))
    print('size: {}'.format(y_pred_tr.sum()))

    X_tran_te = tf.transform(X_test)
    y_proba = bnb.predict_proba(X_tran_te)
    y_pred_te = np.empty_like(y_test)
    for i,row in enumerate(y_pred_te):
        if i in np.where(y_proba[:,1]>.00001)[0]:
            y_pred_te[i] = True
        else:
            y_pred_te[i] = False
    print("test recall:  {}".format(recall_score(y_test,y_pred_te)))
    print("test precision:  {}".format(precision_score(y_test,y_pred_te)))
    print('size: {}'.format(y_pred_te.sum()))
    pass

# pickle models
def pickle_models(tf,bnb):
    with open('tf_org_desc_rec.pickle', 'wb') as handle:
        pickle.dump(tf, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('bnb_org_desc_rec.pickle', 'wb') as handle2:
        pickle.dump(bnb, handle2, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    df = data_to_df()
    unpack_tix(df)
    add_fraud(df)
    X, y = desc_to_nump(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    tf,bnb = build_model(X_train,y_train)
    score_train_test(tf,bnb, X_train, X_test, y_train, y_test)
    pickle_models(tf,bnb)
