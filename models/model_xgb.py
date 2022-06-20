from tkinter import Grid
from lightgbm import early_stopping
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn import preprocessing
import pandas as pd
from preprocess import *


def XGBoost(train, test, submission):

    preprocessing()

    y_train = train['isFraud'].copy()
    # Drop target, fill in NaNs
    X_train = train.drop('isFraud', axis=1)
    # X_train = train.copy()
    X_test = test.copy()

    del train, test

    X_train = X_train.fillna(-999)
    X_test = X_test.fillna(-999)

    # Label Encoding
    # for f in X_train.columns:
    #     if X_train[f].dtype=='object' or X_test[f].dtype=='object': 
    #         lbl = preprocessing.LabelEncoder()
    #         lbl.fit(list(X_train[f].values) + list(X_test[f].values))
    #         X_train[f] = lbl.transform(list(X_train[f].values))
    #         X_test[f] = lbl.transform(list(X_test[f].values))   

    X_train, X_test = encode_FE(X_train, X_test)


    


    # def encode_FE(df1, df2, cols):
    #     for col in cols:
    #         df = pd.concat([df1[col],df2[col]])
    #         vc = df.value_counts(dropna=True, normalize=True).to_dict()
    #         vc[-1] = -1
    #         nm = col+'_FE'
    #         df1[nm] = df1[col].map(vc)
    #         df1[nm] = df1[nm].astype('float32')
    #         df2[nm] = df2[col].map(vc)
    #         df2[nm] = df2[nm].astype('float32')
    #         print(nm,', ',end='')
    #     return df1, df2


    clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=9,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        missing=-999,
        random_state=2019,
        # tree_method='gpu_hist'  # THE MAGICAL PARAMETER
    )

    # params = {'max_depth':[5,7]}

    # gridcv = GridSearchCV(clf, param_grid=params, cv=3)

    # gridcv.fit(X_train,y_train, early_stopping=30, eval_metric='auc', eval_set = [(X_val, y_val)])

    # gridcv = GridSearchCV


    clf.fit(X_train, y_train)


    submission['isFraud'] = clf.predict_proba(X_test)[:,1]
    submission.to_csv('uid+clss_xgb.csv')