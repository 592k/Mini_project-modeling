#%%
import pandas as pd
from dataloader import *
from preprocesses import preprocess_uid, preprocess_V, preprocess_C, preprocess_classreduce, preprocess_dropcol
from models import model_xgb
# from preprocesses.preprocess_kwon import *

def preprocessing():
    train_data, test_data = load_data()

    train_data, test_data = preprocess_classreduce.classreduce(train_data, test_data)
    train_data, test_data = preprocess_uid.pre_uid(train_data, test_data)  
    train_data, test_data = preprocess_V.pre_V(train_data, test_data)
    train_data = preprocess_C.preprocess_C(train_data)
    # train_data, test_data = preprocess_dropcol.dropcol(train_data, test_data)

    def encode_FE(df1, df2):
        for col in df1.columns:
            df = pd.concat([df1[col],df2[col]])
            vc = df.value_counts(dropna=True, normalize=True).to_dict()
            vc[-1] = -1
            nm = col
            df1[nm] = df1[col].map(vc)
            df1[nm] = df1[nm].astype('float32')
            df2[nm] = df2[col].map(vc)
            df2[nm] = df2[nm].astype('float32')
            print(nm,', ',end='')
        return df1, df2


#%%
     
#%%
submission = load_submission()


# %%
