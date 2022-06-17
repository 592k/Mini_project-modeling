#%%
from dataloader import *
from preprocesses.preprocess_V import *
# from preprocesses.preprocess_kwon import *


train_data, test_data = load_data()  
train_data, test_data = pre_V(train_data, test_data)
# train_data, test_data = kwon(train_data, test_data)

# %%