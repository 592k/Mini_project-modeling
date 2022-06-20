from preprocess import preprocessing
from dataloader import load_submission
from models.model_xgb import XGBoost
import datetime
# from utils import load_config

def main():

    X, y, test_data = preprocessing()
    submission = load_submission()

    submission['isFraud'] = XGBoost(X, y, test_data) # models

    submission.to_csv(f'submission{datetime.datetime.now().strftime("%m-%d %H:%M")}.csv')

    print('Done')

if __name__ == "__main__":

    main()
