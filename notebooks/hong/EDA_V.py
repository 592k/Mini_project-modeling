#%%
base = './ieee-fraud-detection/'

import imp
import time
import os
import numpy as np
import pandas as pd
from pyparsing import line
import xgboost as xgb
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
print("XGBoost version:", xgb.__version__)
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
# scaler = StandardScaler()
#%%
import matplotlib.pyplot as plt
import seaborn as sns

#%%

train_transaction= pd.read_csv(base + 'train_transaction.csv',index_col='TransactionID')
train_identity = pd.read_csv(base +'train_identity.csv',index_col='TransactionID')

test_transaction= pd.read_csv(base + 'test_transaction.csv',index_col='TransactionID')
test_identity = pd.read_csv(base + 'test_identity.csv',index_col='TransactionID')
submission = pd.read_csv(base + 'sample_submission.csv',index_col='TransactionID')

fix = {o:n for o, n in zip(test_identity.columns, train_identity.columns)}
test_identity.rename(columns=fix, inplace=True)

train_merge = train_transaction.merge(train_identity, how='left', on='TransactionID')
test_merge = test_transaction.merge(test_identity, how='left', on='TransactionID')

# train data 타겟과 분리
X = train_merge.drop(columns=['isFraud'], axis=1)
y = train_merge['isFraud'].copy()
del train_merge

# V만 있는 데이터프레임 생성, 본 데이터프레임에서는 drop
X_V = X.loc[:,X.columns.str.contains('V')]
X = X.drop(X.loc[:,X.columns.str.contains('V')].columns, axis=1)

test_V = test_merge.loc[:,test_merge.columns.str.contains('V')]
test_merge = test_merge.drop(test_merge.loc[:,test_merge.columns.str.contains('V')].columns, axis=1)

#%%
# V컬럼을 결측치수별로 분류해서 묶어주고, 묶음마다 평균 분산을 구함
# 분산이 큰 묶음들은 배제하고 분산이 낮은 묶음들만 비교, 9번째 '168969'가 가장 높은 스코어로 확인됨
# 해당 묶음만 PCA후 본 데이터프레임에 merge, 나머지 V컬럼들은 모두 drop -> preprocess_V
nan_counts = list(X_V.isnull().sum().value_counts().index)
temp = []
for i in nan_counts:
    temp.append(format(np.var(X_V.loc[:,X_V.isnull().sum() == i]).mean(), '.2f'))
pd.DataFrame(temp, index=nan_counts)
# %%