import numpy as np
  
def get_oof_ypred(model, x_val, x_test, modelname="lgb",):  
    """
    get oof and target predictions
    """
    sklearns = ["xgb", "catb", ]

    # sklearn API
    if modelname in sklearns:
        oof_pred = model.predict_proba(x_val)
        y_pred = model.predict_proba(x_test)
        oof_pred = oof_pred[:, 1]
        y_pred = y_pred[:, 1]
    else:
        oof_pred = model.predict(x_val)
        y_pred = model.predict(x_test)

    return oof_pred, y_pred