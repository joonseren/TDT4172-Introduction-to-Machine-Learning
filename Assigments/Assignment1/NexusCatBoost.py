# nexus_catboost.py
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

class NexusCatBoost:
    def __init__(self, cat_cols=None, exclude=None, **kwargs):
        # Kun strengt nødvendig: cat_cols, exclude og noen fornuftige defaults
        self.cat_cols = list(cat_cols) if cat_cols else []
        self.exclude = set(exclude) if exclude else set()
        # Faste, fornuftige defaults – kan overstyres via **kwargs
        params = dict(
            iterations=1500,
            learning_rate=0.05,
            depth=6,
            loss_function="RMSE",
            random_seed=42,
            verbose=False
        )
        params.update(kwargs)
        self.model = CatBoostRegressor(**params)
        self.columns_ = None
        self.cat_idx_ = None

    def _prepare(self, X: pd.DataFrame):
        # Dropp ekskluderte og behold kolonne-rekkefølge
        Xp = X.drop(columns=[c for c in self.exclude if c in X.columns], errors="ignore").copy()
        # Lagre/bruk samme kolonnerekkefølge som ved fit
        if self.columns_ is None:
            self.columns_ = list(Xp.columns)
        Xp = Xp.reindex(columns=self.columns_)
        # Finn indeksene til kategoriske kolonner som faktisk finnes
        self.cat_idx_ = [i for i, c in enumerate(self.columns_) if c in self.cat_cols]
        return Xp

    def fit(self, X: pd.DataFrame, y):
        # Forvent at y allerede er log1p-transformert utenfor klassen
        Xp = self._prepare(X)
        pool = Pool(Xp, y, cat_features=self.cat_idx_)
        self.model.fit(pool)
        return self

    def predict(self, X: pd.DataFrame):
        Xp = self._prepare(X)
        pool = Pool(Xp, cat_features=self.cat_idx_)
        # Returnerer prediksjoner i SAMME skala som y under fit (dvs. log-space hvis du ga log1p)
        return self.model.predict(pool)