import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier

def catb_model(cls, train_set, val_set):
    """
    CatBoost hyperparameters and models
    """

    # verbose
    verbosity = 10000 if cls.verbose else 0

    # list is here: https://catboost.ai/docs/concepts/python-reference_parameters-list.html
    params = { 'task_type': "CPU",
                'learning_rate': 0.008, 
                'iterations': 16000,
                'colsample_bylevel': 0.44,
                'random_seed': cls.seed,
                'use_best_model': True,
                'early_stopping_rounds': 80
                }
    params["loss_function"] = "Logloss"
    params["eval_metric"] = "Logloss"
    params['class_weights'] = [5, 100]

    # modeling
    model = CatBoostClassifier(**params)
    model.fit(train_set['X'], train_set['y'], eval_set=(val_set['X'], val_set['y']),
        verbose=verbosity, cat_features=cls.categoricals)
    
    # feature importance
    fi = model.get_feature_importance()

    return model, fi
