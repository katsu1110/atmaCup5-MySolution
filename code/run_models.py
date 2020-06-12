import numpy as np
import pandas as pd
import os, sys
from typing import List, NoReturn, Union, Tuple, Optional, Text, Generic, Callable, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, QuantileTransformer
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, mean_squared_error, mean_absolute_error, f1_score
import lightgbm as lgb

# visualize
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
from matplotlib import pyplot
from matplotlib.ticker import ScalarFormatter
sns.set_context("talk")
style.use('seaborn-colorblind')

# gbdts
from get_pred import get_oof_ypred
from xgb_param_models import xgb_model
from lgb_param_models import lgb_model
from catb_param_models import catb_model

# tricks
UNDERSAMPLING_BAGGING = True
GEOMEAN = True
N_BAG = 10
N_UNDER = 9

class RunModel(object):
    """
    Model Fitting and Prediction Class:

    :INPUTS:

    :train_df: train pandas dataframe
    :test_df: test pandas dataframe
    :target: target column name (str)
    :features: list of feature names
    :categoricals: list of categorical feature names. Note that categoricals need to be in 'features'
    :model: 'lgb', 'xgb', 'catb', 'linear', or 'nn'
    :task: 'regression', 'multiclass', or 'binary'
    :n_splits: K in KFold (default is 4)
    :cv_method: 'KFold', 'StratifiedKFold', 'TimeSeriesSplit', 'GroupKFold', 'StratifiedGroupKFold'
    :group: group feature name when GroupKFold or StratifiedGroupKFold are used
    :parameter_tuning: True or False (not implemented for now)
    :seed: seed (int)
    :scaler: None, 'MinMax', 'Standard'
    :verbose: bool

    :EXAMPLE:

    # fit LGB regression model
    model = RunModel(train_df, test_df, target, features, categoricals=categoricals,
            model="lgb", task="regression", n_splits=4, cv_method="KFold", 
            group=None, seed=1220, scaler=None)
    
    # save predictions on train, test data
    np.save("y_pred", model.y_pred)
    np.save("oof", model.oof)
    """

    def __init__(self, train_df : pd.DataFrame, test_df : pd.DataFrame, target : str, features : List, categoricals: List=[],
                model : str="lgb", task : str="regression", n_splits : int=4, cv_method : str="KFold", 
                group : str=None, parameter_tuning=False, seed : int=1220, scaler : str=None, verbose=True):

        # display info
        print("##############################")
        print(f"Starting training model {model} for a {task} task:")
        print(f"- train records: {len(train_df)}, test records: {len(test_df)}")
        print(f"- target column is {target}")
        print(f"- {len(features)} features with {len(categoricals)} categorical features")
        print(f"- CV strategy : {cv_method} with {n_splits} splits")
        if group is None:
            print(f"- no group parameter is used for validation")
        else:
            print(f"- {group} as group parameter")
        if scaler is None:
            print("- No scaler is used")
        else:
            print(f"- {scaler} scaler is used")
        print("##############################")

        # class initializing setups
        self.train_df = train_df
        self.test_df = test_df
        self.target = target
        self.features = features
        self.categoricals = categoricals
        self.model = model
        self.task = task
        self.n_splits = n_splits
        self.cv_method = cv_method
        self.group = group
        self.parameter_tuning = parameter_tuning
        self.seed = seed
        self.scaler = scaler
        self.verbose = verbose
        self.cv = self.get_cv()
        self.y_pred, self.score, self.model, self.oof, self.y_val, self.fi_df = self.fit()

    def train_model(self, train_set, val_set):
        """
        employ a model
        """
        # compile model
        if self.model == "lgb": # LGB             
            model, fi = lgb_model(self, train_set, val_set)

        elif self.model == "xgb": # xgb
            model, fi = xgb_model(self, train_set, val_set)

        elif self.model == "catb": # catboost
            model, fi = catb_model(self, train_set, val_set)
        
        return model, fi # fitted model and feature importance

    def convert_dataset(self, x_train, y_train, x_val, y_val):
        """
        dataset converter
        """
        if (self.model == "lgb") & (self.task != "multiclass"):
            train_set = lgb.Dataset(x_train, y_train, categorical_feature=self.categoricals)
            val_set = lgb.Dataset(x_val, y_val, categorical_feature=self.categoricals)
        else:
            train_set = {'X': x_train, 'y': y_train}
            val_set = {'X': x_val, 'y': y_val}
        return train_set, val_set

    def calc_metric(self, y_true, y_pred): 
        """
        calculate evaluation metric for each task
        this may need to be changed based on the metric of interest
        """
        return metrics.average_precision_score(y_true, y_pred) # log_loss

    def get_cv(self):
        """
        employ CV strategy
        """

        # return cv.split
        if self.cv_method == "KFold":
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            return cv.split(self.train_df)
        elif self.cv_method == "StratifiedKFold":
            cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            return cv.split(self.train_df, self.train_df[self.target])
        
    def fit(self):
        """
        perform model fitting        
        """

        # initialize
        y_vals = np.zeros((self.train_df.shape[0], ))
        oof_pred = np.zeros((self.train_df.shape[0], ))
        if GEOMEAN:
            y_pred = np.ones((self.test_df.shape[0], ))
        else:
            y_pred = np.zeros((self.test_df.shape[0], ))

        # group does not kick in when group k fold is used
        if self.group is not None:
            if self.group in self.features:
                self.features.remove(self.group)
            if self.group in self.categoricals:
                self.categoricals.remove(self.group)
        fi = np.zeros((self.n_splits, len(self.features)))

        # fitting with out of fold
        x_test = self.test_df[self.features]
        for fold, (train_idx, val_idx) in enumerate(self.cv):
            # train test split
            x_train, x_val = self.train_df.loc[train_idx, self.features], self.train_df.loc[val_idx, self.features]
            y_train, y_val = self.train_df.loc[train_idx, self.target], self.train_df.loc[val_idx, self.target]
            
            ######## TRICKS ########## 
            if UNDERSAMPLING_BAGGING:
                # bagging
                if GEOMEAN:
                    oofs = np.ones(oof_pred[val_idx].shape)
                    ypred = np.ones(y_pred.shape)
                else:
                    oofs = np.zeros(oof_pred[val_idx].shape)
                    ypred = np.zeros(y_pred.shape)
                for n in range(N_BAG):
                    # undersampling
                    pos_ratio = np.sum(y_train == 1) / y_train.shape[0]
                    under_df = self.train_df.loc[train_idx, :].reset_index(drop=True)
                    under_df1 = under_df.loc[under_df[self.target] == 1, :]
                    poslen = len(under_df1)
                    under_df0 = under_df.loc[under_df[self.target] == 0, :].sample(N_UNDER * poslen, random_state=self.seed+fold+n+1)
                    under_df = pd.concat([under_df1, under_df0], ignore_index=True)
                    x_train = under_df[self.features]
                    y_train = under_df[self.target]
                    print("UNDERSAMPLING+BAGGING {}: positive ratio = {:.4f} to {:.4f}".format(n, pos_ratio, np.sum(y_train == 1) / y_train.shape[0]))
                    
                    # model fitting
                    train_set, val_set = self.convert_dataset(x_train, y_train, x_val, y_val)
                    model, importance = self.train_model(train_set, val_set)
                
                    # predictions and check cv score
                    oofs_tmp, ypred_tmp = get_oof_ypred(model, x_val, x_test, self.model)
                    if GEOMEAN:
                        oofs *= oofs_tmp
                        ypred *= ypred_tmp
                    else:
                        oofs += oofs_tmp / N_BAG
                        ypred += ypred_tmp / N_BAG
                if GEOMEAN:
                    oofs = oofs ** (1 / N_BAG)
                    ypred = ypred ** (1 / N_BAG)
            else:
                           
                # model fitting
                train_set, val_set = self.convert_dataset(x_train, y_train, x_val, y_val)
                model, importance = self.train_model(train_set, val_set)

                # predictions and check cv score
                oofs, ypred = get_oof_ypred(model, x_val, x_test, self.model)
              
            fi[fold, :] = importance
            y_vals[val_idx] = y_val
            if GEOMEAN:
                y_pred *= ypred.reshape(y_pred.shape)
            else:
                y_pred += ypred.reshape(y_pred.shape) / self.n_splits
            oof_pred[val_idx] = oofs.reshape(oof_pred[val_idx].shape)
            print('Partial score of fold {} is: {}'.format(fold, self.calc_metric(y_vals[val_idx], 
                oof_pred[val_idx])))

        # feature importance data frame
        fi_df = pd.DataFrame()
        for n in np.arange(self.n_splits):
            tmp = pd.DataFrame()
            tmp["features"] = self.features
            tmp["importance"] = fi[n, :]
            tmp["fold"] = n
            fi_df = pd.concat([fi_df, tmp], ignore_index=True)
        gfi = fi_df[["features", "importance"]].groupby(["features"]).mean().reset_index()
        fi_df = fi_df.merge(gfi, on="features", how="left", suffixes=('', '_mean'))

        # outputs
        if GEOMEAN:
            y_pred = y_pred ** (1 / self.n_splits)
        loss_score = self.calc_metric(y_vals, oof_pred)

        if self.verbose:
            print('Our oof loss score is: ', loss_score)
        return y_pred, loss_score, model, oof_pred, y_vals, fi_df

    def plot_feature_importance(self, rank_range=[1, 50]):
        """
        function for plotting feature importance 

        :EXAMPLE:
        # fit LGB regression model
        model = RunModel(train_df, test_df, target, features, categoricals=categoricals,
                model="lgb", task="regression", n_splits=4, cv_method="KFold", 
                group=None, seed=1220, scaler=None)
        
        # plot 
        fi_df = model.plot_feature_importance(rank_range=[1, 100])
        
        """
        # plot feature importance
        _, ax = plt.subplots(1, 1, figsize=(10, 20))
        sorted_df = self.fi_df.sort_values(by = "importance_mean", ascending=False).reset_index().iloc[self.n_splits * (rank_range[0]-1) : self.n_splits * rank_range[1]]
        sns.barplot(data=sorted_df, x ="importance", y ="features", orient='h')
        ax.set_xlabel("feature importance")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        return sorted_df