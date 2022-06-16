#%%
from dataloader import *
from preprocesses.preprocess_V import *

X, y, test_merge = load_data()
X, test_merge = pre_V(X, test_merge)
# %%