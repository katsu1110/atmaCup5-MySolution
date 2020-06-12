import numpy as np
import pandas as pd
import lightgbm as lgb

def lgb_model(cls, train_set, val_set):
    """
    LightGBM hyperparameters and models
    """

    # verbose
    verbosity = 10000 if cls.verbose else 0

    # list is here: https://lightgbm.readthedocs.io/en/latest/Parameters.html
    if cls.seed == 116:
        params = {
        'num_leaves': 202,
        'max_depth': 12,
        'min_child_weight': 6,
        'feature_fraction': 0.6043728425132923,
        'bagging_fraction': 0.863049104308609,
        'bagging_freq': 8,
        'min_child_samples': 80,
        'lambda_l1': 0.0014892942087737318,
        'lambda_l2': 0.0012196126344416074
            }
    elif cls.seed == 1220:
        params = {'num_leaves': 207, 'max_depth': 11, 'min_child_weight': 6, 'feature_fraction': 0.6057502186099188, 'bagging_fraction': 0.6138865979012038, 'bagging_freq': 4, 'min_child_samples': 80, 'lambda_l1': 0.10129229099955826, 'lambda_l2': 0.003039240025379999}
    elif cls.seed == 217:
        params = {'num_leaves': 145, 'max_depth': 13, 'min_child_weight': 3, 'feature_fraction': 0.5171179643463687, 'bagging_fraction': 0.47423149034800405, 'bagging_freq': 8, 'min_child_samples': 10, 'lambda_l1': 7.61128642340862e-07, 'lambda_l2': 0.0008327779815476518}
    elif cls.seed == 528:
        params = {'num_leaves': 408, 'max_depth': 9, 'min_child_weight': 1, 'feature_fraction': 0.6921507084993932, 'bagging_fraction': 0.5565900547808434, 'bagging_freq': 3, 'min_child_samples': 72, 'lambda_l1': 0.0004813686638832776, 'lambda_l2': 0.015485342108910236}
    elif cls.seed == 427:
        params = {'num_leaves': 622, 'max_depth': 5, 'min_child_weight': 4, 'feature_fraction': 0.502531967252501, 'bagging_fraction': 0.858071478559012, 'bagging_freq': 2, 'min_child_samples': 32, 'lambda_l1': 0.012487519569886552, 'lambda_l2': 0.062022000785908436}
    params["metric"] = "binary_logloss" # other candidates: binary_logloss
    params["is_unbalance"] = True # assume unbalanced data
    params['n_estimators'] = 8000
    params['objective'] = cls.task
    params['learning_rate'] = 0.008
    params['seed'] = cls.seed
    params['early_stopping_rounds'] = 80
    params['num_boost_round'] = 8000
    
    # modeling and feature importance
    model = lgb.train(params, train_set, valid_sets=[train_set, val_set], verbose_eval=verbosity)
    fi = model.feature_importance(importance_type="gain")

    return model, fi
