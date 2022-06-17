import pandas as pd

def dropcol(df_train, df_test):

    columns = ['id_07', 'id_08', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_33',
            'R_emaildomain', 
            'dist2', 
            'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C10', 'C11', 'C12', 'C13', 'C14',
            'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
            'addr1', 'addr2',
            'dist1',
            'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 
            ]

    for column in columns:
        df_train.drop(columns=[column], inplace=True)
        df_test.drop(columns=[column], inplace=True)
    
    return df_train, df_test