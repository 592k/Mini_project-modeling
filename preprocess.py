#%%
import pandas as pd
from dataloader import *
from preprocesses import preprocess_uid, preprocess_V, preprocess_C, preprocess_classreduce, preprocess_dropcol
from models import model_xgb
# from preprocesses.preprocess_kwon import *

train_data, test_data = load_data()

train_data, test_data = preprocess_classreduce.classreduce(train_data, test_data)
train_data, test_data = preprocess_uid.pre_uid(train_data, test_data)  
train_data, test_data = preprocess_V.pre_V(train_data, test_data)
train_data = preprocess_C.preprocess_C(train_data)
# train_data, test_data = preprocess_dropcol.dropcol(train_data, test_data)




#%%
     
#%%
submission = load_submission()

model_xgb.XGBoost(train_data, test_data, submission)


# %%
