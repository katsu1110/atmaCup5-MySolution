"""
perform median ensemble

:USAGE:
>>> python ensemble.py

"""

### Libraries ###
import numpy as np
import pandas as pd
import os, sys
import math
from typing import List, NoReturn, Union, Tuple, Optional, Text, Generic, Callable, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, QuantileTransformer
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn import metrics

### data location ###
PSEUDO = int(sys.argv[1])
input_path = "../input/"
output_path = "../output/"

### load data ###
oofs = {
    'lgb': np.load(output_path + f'oof_lgb{PSEUDO}.npy'),
    'catb': np.load(output_path + f'oof_catb{PSEUDO}.npy'),
    'xgb': np.load(output_path + f'oof_xgb{PSEUDO}.npy'),
}
y_preds = {
    'lgb': np.load(output_path + f'ypred_lgb{PSEUDO}.npy'),
    'catb': np.load(output_path + f'ypred_catb{PSEUDO}.npy'),
    'xgb': np.load(output_path + f'ypred_xgb{PSEUDO}.npy'),
}
train = pd.read_csv(input_path + 'train.csv')
sub = pd.read_csv(input_path + "atmaCup5__sample_submission.csv")
label = train['target'].values
orig_trlen = train.shape[0]

### median ensemble ###
oof_ = pd.DataFrame()
ypred_ = pd.DataFrame()
for m in oofs.keys():
    oof_[m] = oofs[m]
    ypred_[m] = y_preds[m]
oof_pred_med = oof_.median(axis=1)
y_pred_med = ypred_.median(axis=1)

### submission ###
sub['target'] = y_pred_med
sub.to_csv(output_path + f'submission_med{PSEUDO}.csv', index=False)
np.save(output_path + 'oof_med', oof_pred_med)
np.save(output_path + 'ypred_med', y_pred_med)
print("submitted!")
