import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import random
import pickle
random.seed(1337)

# load pickled dataframe
def load_data(filepath):
    return pd.read_pickle(filepath)

# random forest classifier with fraud as y and features as X
def run_random_forest(df):
    y = df.pop('fraud')
    X = df
    # X = df.drop('fraud', 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=29)
    clf = RandomForestClassifier(n_jobs=2,random_state=21)
    features = X.columns
    # clf = Pipeline([("imputer", Imputer(missing_values=0, strategy="mean", axis=0)), ("forest", RandomForestClassifier(n_jobs=2))])

    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    feature_importances = list(zip(X.columns, clf.feature_importances_))
    print(feature_importances)
    print(classification_report(y_test, predictions))
    return clf

# pickle models
def pickle_models(clf):
    with open('pickles/randomforest_fraud.pickle', 'wb') as handle:
        pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    df = load_data('pickles/df.pkl')
    clf = run_random_forest(df)
    pickle_models(clf)
