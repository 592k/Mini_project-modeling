import pandas as pd

def pre_C(X, test_merge):

    X.drop()
    for i in ['C1','C9']:
        
        fenc = X[i].quantile(.75) + (3 * (X[i].quantile(.75) - X[i].quantile(.25)))
        X.drop(X[i][X[i]>fenc].index, inplace=True)
