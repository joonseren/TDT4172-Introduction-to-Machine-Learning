import numpy as np
from xgboost import XGBRegressor

class NexusXGB:
    def __init__(self, **params):
        # default params, kan overskrives av brukeren
        defaults = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1
        }
        defaults.update(params)
        self.model = XGBRegressor(**defaults)

    def fit(self, X, y):
        # XGBoost liker log-transformert target når vi evaluerer med RMSLE
        self.y_shift = 1e-9  # for å unngå log(0)
        y_log = np.log1p(y)
        self.model.fit(X, y_log)

    def predict(self, X):
        y_log_pred = self.model.predict(X)
        return np.expm1(y_log_pred)  # inverter log1p