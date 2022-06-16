#%%
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

#%%
def pre_V(X, test_merge):
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    X_V = X.loc[:,X.columns.str.contains('V')]
    X = X.drop(X.loc[:,X.columns.str.contains('V')].columns, axis=1)

    test_V = test_merge.loc[:,test_merge.columns.str.contains('V')]
    test_merge = test_merge.drop(test_merge.loc[:,test_merge.columns.str.contains('V')].columns, axis=1)

    same_nan_columns = list(X_V.isnull().sum().value_counts())
    nan_counts = list(X_V.isnull().sum().value_counts().index)
    X2 = X
    for i,j in zip(nan_counts, same_nan_columns):
        if nan_counts.index(i) == 8:
            T = X_V.loc[:,X_V.isnull().sum() == i].dropna(axis=0)
            temp_index = T.index
            Z = scaler.fit_transform(T)
            pca = PCA(n_components=1)
            ratio =  pd.DataFrame(pca.fit_transform(Z), columns=[f'V{j}'],index = temp_index)
            X2 = X2.merge(ratio, how="left", on='TransactionID')
        
    same_nan_columns = list(test_V.isnull().sum().value_counts())
    nan_counts = list(test_V.isnull().sum().value_counts().index)
    test_merge2 = test_merge
    for i,j in zip(nan_counts, same_nan_columns):
        if nan_counts.index(i) == 8:
            T = test_V.loc[:,test_V.isnull().sum() == i].dropna(axis=0)
            temp_index = T.index
            Z = scaler.fit_transform(T)
            pca = PCA(n_components=1)
            ratio =  pd.DataFrame(pca.fit_transform(Z), columns=[f'V{j}'],index = temp_index)
            test_merge2 = test_merge2.merge(ratio, how="left", on='TransactionID')

    for f in X2.columns:
        if X2[f].dtype=='object' or test_merge2[f].dtype=='object' : 
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(X2[f].values) + list(test_merge2[f].values))
            X2[f] = lbl.transform(list(X2[f].values))
            test_merge2[f] = lbl.transform(list(test_merge2[f].values))
            
    return(X2, test_merge2)