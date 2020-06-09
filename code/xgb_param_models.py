import numpy as np
import pandas as pd
import xgboost as xgb
import operator

def xgb_model(cls, train_set, val_set):
    """
    XGB hyperparameters and models
    """

    # verbose
    verbosity = 10000 if cls.verbose else 0

    # list is here: https://xgboost.readthedocs.io/en/latest/parameter.html
    params = {
        'colsample_bytree': 0.56,                 
        'learning_rate': 0.008,
        'max_depth': 9,
        'subsample': 1,
        'min_child_weight': 4,
        'gamma': 0.24,
        'alpha': 0,
        'lambda': 1,
        'seed': cls.seed,
        'n_estimators': 8000
    }
    params["objective"] = 'binary:logistic'
    params["eval_metric"] = "logloss"
    params['scale_pos_weight'] = 20

    # modeling
    model = xgb.XGBClassifier(**params)
    model.fit(train_set['X'], train_set['y'], eval_set=[(val_set['X'], val_set['y'])],
                    early_stopping_rounds=80, verbose=verbosity)

    # feature importance
    importance = model.get_booster().get_score(importance_type='gain')
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    fi = np.zeros(len(cls.features))
    for i, f in enumerate(cls.features):
        try:
            fi[i] = df.loc[df['feature'] == f, "fscore"].iloc[0]
        except: # ignored by XGB
            continue

    return model, fi
