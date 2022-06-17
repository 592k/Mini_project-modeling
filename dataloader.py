import pandas as pd


def load_data():
    base = './ieee-fraud-detection/'
    train_transaction= pd.read_csv(base + 'train_transaction.csv',index_col='TransactionID')
    train_identity = pd.read_csv(base +'train_identity.csv',index_col='TransactionID')

    test_transaction= pd.read_csv(base + 'test_transaction.csv',index_col='TransactionID')
    test_identity = pd.read_csv(base + 'test_identity.csv',index_col='TransactionID')

    fix = {o:n for o, n in zip(test_identity.columns, train_identity.columns)}
    test_identity.rename(columns=fix, inplace=True)

    train_merge = train_transaction.merge(train_identity, how='left', on='TransactionID')
    test_merge = test_transaction.merge(test_identity, how='left', on='TransactionID')

    # X = train_merge.drop(columns=['isFraud'], axis=1)
    # y = train_merge['isFraud'].copy()
    # del train_merge
    
    return train_merge, test_merge

def load_submission():
    base = './ieee-fraud-detection/'
    submission = pd.read_csv(base + 'sample_submission.csv',index_col='TransactionID')
    return submission

