import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


class DecisionTreeMission3:
    def __init__(self, decision_tree_params=None):
        self.decision_tree_params = decision_tree_params or {} 
        self.model = None

    def fit(self, X, y):
        self.model = DecisionTreeClassifier(**self.decision_tree_params)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def roc_metrics(self, X, y):
        p = self.predict_proba(X)
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        return tpr, fpr, auc  
    
    def decryption(self, X, feature_idx=3):
        # NB: +1 før % 2, slik som i løsningen du viste
        X_new = X.copy()
        X_new[:, feature_idx] = (X_new[:, feature_idx] * 1000 + 1) % 2
        return X_new