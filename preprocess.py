#%%
import pandas as pd
from dataloader import *
from preprocesses import preprocess_uid, preprocess_V, preprocess_C, preprocess_classreduce, preprocess_dropcol
from models import model_xgb
# from preprocesses.preprocess_kwon import *

def preprocessing():

    def encode_FE(train_data, test_data):
        for col in train_data.columns:
            df = pd.concat([train_data[col],test_data[col]])
            vc = df.value_counts(dropna=True, normalize=True).to_dict()
            vc[-1] = -1
            nm = col
            train_data[nm] = train_data[col].map(vc)
            train_data[nm] = train_data[nm].astype('float32')
            test_data[nm] = test_data[col].map(vc)
            test_data[nm] = test_data[nm].astype('float32')
            print(nm,', ',end='')
        return train_data, test_data
    
    ##def Label Encoding():
    #   for f in test_data.columns:
    #       if X_train[f].dtype=='object' or X_test[f].dtype=='object': 
    #           lbl = preprocessing.LabelEncoder()
    #           lbl.fit(list(X_train[f].values) + list(X_test[f].values))
    #           X_train[f] = lbl.transform(list(X_train[f].values))
    #           X_test[f] = lbl.transform(list(X_test[f].values))   


    train_data, test_data = load_data()

    train_data, test_data = preprocess_classreduce.classreduce(train_data, test_data)
    
    train_data, test_data = preprocess_uid.pre_uid(train_data, test_data)  
    
    train_data, test_data = preprocess_V.pre_V(train_data, test_data)
    
    train_data = preprocess_C.pre_C(train_data)
    
    train_data, test_data = encode_FE(train_data, test_data)

    train_data, test_data = preprocess_dropcol.dropcol(train_data, test_data)

    X = train_data.drop('isFraud', axis=1)
    y = train_data['isFraud']

    X = X.fillna(-999)
    test_data = test_data.fillna(-999)

    return X, y, test_data



# %%
