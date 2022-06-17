import pandas as pd

def dropcol(df_train):

    columns = ['id_07', 'id_08', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_33',
            'R_emaildomain', 
            'dist2', 
            'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C10', 'C11', 'C12', 'C13', 'C14', 
            ]

    for column in columns:
        df_train.drop(columns=[column], inplace=True)
    
    return df_train