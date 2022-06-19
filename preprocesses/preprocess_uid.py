import pandas as pd

def pre_uid(train, test):

    for df in [train, test]:
        for col in ['D1']:
            df[col+'_shift'] = (df[col] - df.TransactionDT // 24 // 3600).fillna(-400)


    for df in [train, test]:
        df['uid'] = df['D1_shift'].astype(str)+'_'+df['card1'].astype(str)+'_'+df['P_emaildomain'].astype(str)+df['addr1'].astype(str)+'_'+df['ProductCD'].astype(str)

    train, test = encode_FE(train, test, ['uid'])
    test.drop('uid', axis=1)
    train.drop('uid', axis=1)
    return train, test


def encode_FE(df1, df2, cols):
    for col in cols:
        df = pd.concat([df1[col],df2[col]])
        vc = df.value_counts(dropna=True, normalize=True).to_dict()
        vc[-1] = -1
        nm = col+'_FE'
        df1[nm] = df1[col].map(vc)
        df1[nm] = df1[nm].astype('float32')
        df2[nm] = df2[col].map(vc)
        df2[nm] = df2[nm].astype('float32')
        print(nm,', ',end='')
    return df1, df2
