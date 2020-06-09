import numpy as np
import pandas as pd
import lightgbm as lgb

def lgb_model(cls, train_set, val_set):
    """
    LightGBM hyperparameters and models
    """

    # verbose
    verbosity = 100 if cls.verbose else 0

    # list is here: https://lightgbm.readthedocs.io/en/latest/Parameters.html
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
