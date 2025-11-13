import pandas as pd
from sklearn.impute import SimpleImputer

class NexusPreprocessor:
    def __init__(self, cat_cols=None, exclude=None, one_hot=False, normalize_headers=True):
        self.cat_cols = cat_cols if cat_cols is not None else ["ownership_type", "energy_flow_design", "efficiency_grade"]
        self.exclude = set(exclude) if exclude else set()
        self.one_hot = one_hot
        self.normalize_headers = normalize_headers
        self._num_imputer = SimpleImputer(strategy="median")
        self._train_columns = None  # settes når one_hot=True

    # ---------- utils ----------
    def _normalize_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.normalize_headers:
            return df
        df = df.copy()
        df.columns = (df.columns
                      .str.replace("\u200b", "", regex=False)
                      .str.replace("\ufeff", "", regex=False)
                      .str.strip())
        return df

    def _split_cols(self, X: pd.DataFrame):
        cat = [c for c in self.cat_cols if c in X.columns and c not in self.exclude]
        num = [c for c in X.columns if c not in self.exclude and c not in cat]
        return num, cat

    # ---------- NEW: shift-fix for test CSV (kall denne FØR du splitter y/X) ----------
    def fix_shift_on_full_test_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Retter test-sett som er 'shiftet' én kolonne:
        - kol0 (ownership_type) beholdes
        - kol1 (nexus_rating) settes til opprinnelig siste kolonne (grid_connections)
        - øvrige kolonner får verdier fra sin venstre nabo (dvs. vi shifter verdier én til høyre)
        Bevarer kolonnenavn.
        """
        df = self._normalize_cols(df).copy()
        if df.shape[1] < 3:
            return df
        last_vals = df.iloc[:, -1].copy()
        df.iloc[:, 1:] = df.iloc[:, :-1].values
        df.iloc[:, 1] = last_vals.values  # nå er 'nexus_rating' korrekt
        return df

    # ---------- standard preprocess ----------
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._normalize_cols(X).copy()
        X = X.drop(columns=list(self.exclude & set(X.columns)), errors="ignore")
        num_cols, cat_cols = self._split_cols(X)

        if num_cols:
            X[num_cols] = self._num_imputer.fit_transform(X[num_cols])

        for c in cat_cols:
            X[c] = X[c].fillna(-1)

        if self.one_hot and cat_cols:
            X = pd.get_dummies(X, columns=cat_cols, dummy_na=False)
            self._train_columns = X.columns
        else:
            self._train_columns = None

        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._normalize_cols(X).copy()
        X = X.drop(columns=list(self.exclude & set(X.columns)), errors="ignore")
        num_cols, cat_cols = self._split_cols(X)

        if num_cols:
            X[num_cols] = self._num_imputer.transform(X[num_cols])

        for c in cat_cols:
            X[c] = X[c].fillna(-1)

        if self.one_hot and cat_cols:
            X = pd.get_dummies(X, columns=cat_cols, dummy_na=False)
            if self._train_columns is not None:
                X = X.reindex(columns=self._train_columns, fill_value=0)

        return X

    # ---------- NEW: enkel "pipeline" for features ----------
    def run(self, X_train_raw: pd.DataFrame, X_test_raw: pd.DataFrame):
        """
        Full preprocess for features (ikke target):
        - normaliser header, drop exclude, imputer numeriske, fyll kategoriske, ev. one-hot + align
        Bruk fix_shift_on_full_test_df(df) separat på FULLT test-DF (før du splitter y/X) hvis datasettet er shiftet.
        """
        X_train = self.fit_transform(X_train_raw)
        X_test = self.transform(X_test_raw)
        return X_train, X_test