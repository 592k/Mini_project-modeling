import pandas as pd

def kwon():
    
    col_set = set([])
    for i in ['C1','C9']:
        
        fenc = train_t[i].quantile(.75) + (3 * (train_t[i].quantile(.75) - train_t[i].quantile(.25)))
        col_set.update(list(tc[i][tc[i].values > fenc].index))

    train_t.drop(col_set, aixis=0, inplace=True)
