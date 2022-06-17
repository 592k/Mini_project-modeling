import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def pre_V(X, test_merge):
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    X_V = X.loc[:,X.columns.str.contains('V')]
    X = X.drop(X.loc[:,X.columns.str.contains('V')].columns, axis=1)

    test_V = test_merge.loc[:,test_merge.columns.str.contains('V')]
    test_merge = test_merge.drop(test_merge.loc[:,test_merge.columns.str.contains('V')].columns, axis=1)

    nan_counts = list(X_V.isnull().sum().value_counts().index)
    X2 = X
    for i in nan_counts:
       if np.var(X_V.loc[:,X_V.isnull().sum() == i]).mean() < 1:
            T = X_V.loc[:,X_V.isnull().sum() == i].dropna(axis=0)
            temp_index = T.index
            Z = scaler.fit_transform(T)
            pca = PCA(n_components=1)
            ratio =  pd.DataFrame(pca.fit_transform(Z), columns=[f'V{i}'],index = temp_index)
            X2 = X2.merge(ratio, how="left", on='TransactionID')
        
    nan_counts = list(test_V.isnull().sum().value_counts().index)
    test_merge2 = test_merge
    for i in nan_counts:
        if np.var(test_V.loc[:,test_V.isnull().sum() == i]).mean() < 1:
            T = test_V.loc[:,test_V.isnull().sum() == i].dropna(axis=0)
            temp_index = T.index
            Z = scaler.fit_transform(T)
            pca = PCA(n_components=1)
            ratio =  pd.DataFrame(pca.fit_transform(Z), columns=[f'V{i}'],index = temp_index)
            test_merge2 = test_merge2.merge(ratio, how="left", on='TransactionID')
    
    X2 = X2.drop(X2.iloc[:,-5:-3], axis=1)
    test_merge2 = test_merge2.drop(test_merge2.iloc[:,-5:-3], axis=1)
    
    return(X2, test_merge2)