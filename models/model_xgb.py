from tkinter import Grid
from lightgbm import early_stopping
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn import preprocessing
import pandas as pd
from preprocess import *
from utils import load_config

def XGBoost(X, y, test_data):

    clf = xgb.XGBClassifier(

        load_config()['params']
        
        # tree_method='gpu_hist'  # THE MAGICAL PARAMETER
        )

    # params = {'max_depth':[5,7]}

    # gridcv = GridSearchCV(clf, param_grid=params, cv=3)

    # gridcv.fit(X_train,y_train, early_stopping=30, eval_metric='auc', eval_set = [(X_val, y_val)])

    # gridcv = GridSearchCV


    clf.fit(X, y)

    return clf.predict_proba(test_data)[:,1]