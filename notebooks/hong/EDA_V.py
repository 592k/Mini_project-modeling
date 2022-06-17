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





# 여기부터 EDA
# V만 있는 데이터프레임 생성, 본 데이터프레임에서는 drop
X_V = X.loc[:,X.columns.str.contains('V')]
X = X.drop(X.loc[:,X.columns.str.contains('V')].columns, axis=1)

test_V = test_merge.loc[:,test_merge.columns.str.contains('V')]
test_merge = test_merge.drop(test_merge.loc[:,test_merge.columns.str.contains('V')].columns, axis=1)
#%%
# 339개의 컬럼 확인
# nan값이 많아보인다
X_V.head()
#%%
# 컬럼마다 표준편차의 차이가 심해보인다
X_V.describe()

#%%
# 결측치 확인
# 같은 결측치의 개수를 가진 컬럼들이 보인다
X_V.isnull().sum()
#%%
# V의 339개의 컬럼들은 서로 결측치를 공유하며, 그 기준으로 나눠봤을때 15종류로 나뉘게 된다.
# 그렇다면 이러한 기준은 그 컬럼들이 하나의 성격일수 있겠다는 가정을 해본다.
X_V.isnull().sum().unique()
#%%
# 460110개의 결측 row를 공유하는 컬럼들이 가장 많은 46개이다.
X_V.isnull().sum().value_counts()
#%%
# 결측치별 분류상 가장 수가 많은 46개 컬럼들의 표준편차와 분산의 평균
# 결측치도 높고 분산도 커서 활용할수 있을지 의문이 든다.
X_V.loc[:,X_V.isnull().sum() == 460110].describe()
np.var(X_V.loc[:,X_V.isnull().sum() == 460110]).mean()
#%%
# 15종류의 컬럼들의 평균 분산을 알아봤다.
# 평균분산이 낮은 분류들(5,6,7,9,13번째) 위주로 활용해야 할것 같다.
nan_counts = list(X_V.isnull().sum().value_counts().index)
temp = []
for i in nan_counts:
    temp.append(format(np.var(X_V.loc[:,X_V.isnull().sum() == i]).mean(), '.2f'))
pd.DataFrame(temp, index=nan_counts)
# %%
# 같은 결측치를 가진 컬럼들끼리 pca를 해서 각각의 컬럼으로 축소
# 그중에 평균 분산이 1이하인 위의 5개 분류만 활용. 총 5컬럼
nan_counts = list(X_V.isnull().sum().value_counts().index)
ratio = pd.DataFrame(index = X_V.index)

for i in nan_counts:
    if np.var(X_V.loc[:,X_V.isnull().sum() == i]).mean() < 1:
        T = X_V.loc[:,X_V.isnull().sum() == i].dropna(axis=0)
        temp_index = T.index
        Z = scaler.fit_transform(T)
        pca = PCA(n_components=1)
        ratio = ratio.merge(pd.DataFrame(pca.fit_transform(Z), columns=[f'V_{i}'],index = temp_index), how="left", on='TransactionID')
# %%
# 상관계수 확인결과, 1,2,3번컬럼은 상관관계가 높아보인다.
ratio.corr()
# %%
# 1,2번컬럼 삭제
# V에서는 현재의 총 3개 컬럼만 활용하려 한다.
X_V = ratio.drop(ratio.iloc[:,:2], axis=1)
# %%
# %%
