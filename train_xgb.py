from sklearn.externals import joblib
import xgboost as xgb
import numpy as np


X_train = joblib.load('data/X_train.pkl')
y_train = joblib.load('data/y_train.pkl')

# ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
ratio = 1

X_val = joblib.load('data/X_val.pkl')
y_val = joblib.load('data/y_val.pkl')


print "train XGBClassifier..."
xgb = xgb.XGBClassifier(max_depth=20, n_estimators=1000, learning_rate=0.01,
                        colsample_bytree=0.5, min_child_weight=10, scale_pos_weight=ratio)

xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, eval_metric='logloss', verbose=True)

joblib.dump(xgb, 'models/xgb.pkl')
