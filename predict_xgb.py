import pandas as pd
from sklearn.externals import joblib
import sklearn.metrics as metrics


def predict():
    X_val = joblib.load('data/X_val.pkl')
    y_val = joblib.load('data/y_val.pkl')
    # X_val = joblib.load('data/X_train.pkl')
    # y_val = joblib.load('data/y_train.pkl')

    xgb = joblib.load('models/xgb.pkl')

    print "predict XGBClassifier..."
    preds = xgb.predict_proba(X_val)

    log_loss = metrics.log_loss(y_val, preds)
    print "logloss: ", log_loss
    return preds


def make_submit():
    df = pd.read_csv('data/stage1_sample_submission.csv')
    preds = predict()

    df['cancer'] = preds
    df.to_csv('submit.csv', index=False)

if __name__ == '__main__':
    make_submit()



