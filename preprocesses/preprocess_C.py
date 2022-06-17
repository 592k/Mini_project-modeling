import pandas as pd

def pre_C(X):

    # C1, C9 이상치 제거    
    fenc1 = X['C1'].quantile(.75) + (3 * (X['C1'].quantile(.75) - X['C1'].quantile(.25)))
    fenc2 = X['C9'].quantile(.75) + (3 * (X['C9'].quantile(.75) - X['C9'].quantile(.25)))
    X.drop(X['C1'][X['C1']>fenc1].index, inplace=True)
    X.drop(X['C9'][X['C9']>fenc2].index, inplace=True)

    return X