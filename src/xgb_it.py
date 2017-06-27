from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

# load pickled data
def load_data():
    with open('pickles/df.pkl', 'rb') as handle:
        b = pickle.load(handle)
    return b

# create high/med/low split
def add_priority(row):
    # # average cost of 200 is 75%, 475 = 90%, high priority above that for 10% of fraud detected
    ## 0 = not fraud, 1 = low, 2 = med., 3 = high
    if row['fraud']:
        if row['avg_cost'] <= 200:
            return 1
        elif row['avg_cost'] <= 475:
            return 2
        else:
            return 3
    else:
        return 0

if __name__ == '__main__':
    df = load_data()
    # df = df.drop(['org_description_proba_p'],axis=1)
    # df = df.drop(['org_description_proba_r','org_description_proba_p'],axis=1)
    # df = df.drop(['description_pred','org_description_pred_r','org_description_pred_p'],axis=1)
    # df['descrip_proba'] = np.load('data/description_proba.npy')
    # df['org_desc_proba_p'] = np.load('data/org_description_proba_p')
    # df['org_desc_proba_r'] = np.load('data/org_description_proba_r')
    y = df.pop('fraud').values
    X = df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=29)
    # xgb = RandomForestClassifier(n_estimators=10,random_state=2)
    xgb = XGBClassifier()
    xgb.fit(X_train,y_train)
    y_pred = xgb.predict(X_test)
    print("test recall: {}".format(recall_score(y_test,y_pred)))
    print("test precision: {}".format(precision_score(y_test,y_pred)))

    df['fraud'] = y
    # df_f = df.loc[df['new_col']==1]
    # print("all samples: ".format(df['avg_amounts'].describe()))
    # print('just fraud: '.format(df_f['avg_amounts'].describe()))
    # #avg_cost better: 'avg_cost'
    # print("all samples: ".format(df['avg_cost'].describe()))
    # print('just fraud: '.format(df_f['avg_cost'].describe()))
    # # average cost of 200 is 75%, 475 = 90%, high priority above that for 10% of fraud detected
    # # print('percentile at 475: '.format(df_f.loc[df_f['avg_cost']<=475].shape[0] / float(df_f.shape[0])))

    ## 0 = low, 1 = med, 2 = high
    df['priority'] = df.apply(add_priority,axis=1)
