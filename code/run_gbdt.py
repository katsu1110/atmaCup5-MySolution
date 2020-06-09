"""
run GBDT with undersampling+bagging

:INPUT:
- gbdt type : 'lgb', 'xgb', or 'catb'
- is_pseudo : bool

:EXAMPLE:
>>> python run_gbdt.py 'lgb' False

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

# visualize
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
from matplotlib import pyplot
from matplotlib.ticker import ScalarFormatter
sns.set_context("talk")
style.use('seaborn-colorblind')

from run_models import RunModel

### parameters ###
GBDT = sys.argv[1]
PSEUDO = sys.argv[2]
UNDERSAMPLING_BAGGING = True
GEOMEAN = True

PSEUDO_THRESHOLD = 0.99
N_BAG = 10
N_UNDER = 16

### data location ###
input_path = "../input/"
output_path = "../output/"
fig_path = "../figs/"

### load data ###
def read_data():
    # load
    train = pd.read_csv(output_path + "train.csv")
    test = pd.read_csv(output_path + "test.csv")
    submission = pd.read_csv(input_path + "atmaCup5__sample_submission.csv")
    
    # add cnn features
    cnn_tr = np.load(output_path + 'cnn_features_tr.npy')
    cnn_ts = np.load(output_path + 'cnn_features_ts.npy')
    for i in range(2):
        train[f'cnn_pca{i+1}'] = cnn_tr[:train.shape[0], i]
        test[f'cnn_pca{i+1}'] = cnn_ts[:train.shape[0], i]
    return train, test, submission
train, test, sub = read_data()

### fix json error in LGB ###
def fix_jsonerr(df):
    df.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in df.columns]
    return df
train = fix_jsonerr(train)
test = fix_jsonerr(test)
orig_trlen = train.shape[0]

### features to use ###
target = "target"
categoricals = []
group = None
features = ['grad3_max2min',
 'fft_umap2',
 'fft_pca2',
 'grad2_absmin',
 'params2',
 'sig_mean',
 'exc_wl',
 'grad3_absmin',
 'params5_to_posmean',
 'params0',
 'grad2_min',
 'grad_argmax2min',
 'sig_max',
 'sig_skew',
 'params2_to_posmean',
 'params4',
 'params1_4_ratio',
 'grad2_skew',
 'params13_46_ratio',
 'grad_argmin',
 'peak_pos',
 'num_peaks',
 'grad_skew',
 'grad3_max',
 'params3',
 'layout_a',
 'params6',
 'fft_pca1',
 'sig_kurt',
 'sig_max2min',
 'beta',
 'params6_to_posmean',
 'params3_to_posmean',
 'sig_peaknearsum',
 'fft_umap1',
 'fft_50_100',
 'cnn_pca1', 'cnn_pca2'] # null importanceを元にエラバレシfeatures
print(len(features))
print(features)

### pseudolabel ###
if PSEUDO == 1:
    # load pseudo-data
    def load_pseudo():
        oof = np.load(output_path + "oof_med.npy")
        y_pred = np.load(output_path + "ypred_med.npy")
        return oof, y_pred
    oof, y_pred = load_pseudo()
    test_pseudo = test.loc[y_pred > PSEUDO_THRESHOLD, :].reset_index(drop=True)
    test_pseudo[target] = 1
    train = pd.concat([train, test_pseudo], ignore_index=True)
    print("PSEUDOLABEL: train length {:,} to {:,}".format(orig_trlen, len(train)))

### fitting ###
## seed average
seeds = [217, 1220, 528, 116, 427] # テキトーだぜ
len_s = len(seeds)
if GEOMEAN:
    oof = np.ones(train.shape[0])
    y_preds = np.ones(test.shape[0])
else:
    oof = np.zeros(train.shape[0])
    y_preds = np.zeros(test.shape[0])
    
praucs = np.zeros(len_s)
lls = np.zeros(len_s)
for i, s in enumerate(seeds):
    model = RunModel(train, test, target, features, categoricals=categoricals,
                model=GBDT, task="binary", n_splits=5, cv_method="StratifiedKFold", 
                group=group, seed=s, scaler=None)
    if GEOMEAN:
        oof *= model.oof
        y_preds *= model.y_pred
    else:
        oof += model.oof / len_s
        y_preds += model.y_pred / len_s
    
    praucs[i] = model.score
    lls[i] = metrics.log_loss(model.y_val, model.oof)
if GEOMEAN:
    oof = oof ** (1 / len_s)
    y_preds = y_preds ** (1 / len_s)

### CV ###
prauc = metrics.average_precision_score(train[target].values[:orig_trlen], oof[:orig_trlen])
ll = metrics.log_loss(train[target].values[:orig_trlen], oof[:orig_trlen])
lr_precision, lr_recall, _ = metrics.precision_recall_curve(train['target'].values[:orig_trlen], oof[:orig_trlen])
print(f"Overall PR-AUC = {prauc}, LogLoss = {ll}")
print(metrics.classification_report(y_true=train['target'].values[:orig_trlen], y_pred=np.round(oof[:orig_trlen])))

### plot feature importance ###
if PSEUDO == 0:
    fi_df = model.plot_feature_importance()
    plt.savefig(f'feature_importance_{GBDT}.png', bbox_inches='tight')
    fi_df = fi_df[["features", "importance_mean"]].drop_duplicates().reset_index(drop=True)
    fi_df.to_csv(output_path + f'feature_importance_{GBDT}.csv', index=False)

### save ###
sub["target"] = y_preds
sub.to_csv(output_path + f'submission_{GBDT}.csv', index=False)
print("submitted!")
np.save(output_path + f"oof_{GBDT}", oof)
np.save(output_path + f"ypred_{GBDT}", y_preds)
print("saved!")
