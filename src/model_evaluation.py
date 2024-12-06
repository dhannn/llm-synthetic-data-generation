from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

import numpy as np
import pandas as pd

class ModelEvaluator:

    model: BaseEstimator
    pipeline: Pipeline
    kf: KFold

    def __init__(self, model: BaseEstimator, k=6, random_state=None) -> None:
        self.model = model
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer()),
            ('classifier', model)    
        ])

        if random_state != None:
            np.random.seed(random_state)
            self.kf = KFold(n_splits=k, shuffle=True, random_state=42)

    def evaluate(self, X: pd.DataFrame, y: pd.Series, has_synthetic=False):
        # return cross_val_score(self.pipeline, X, y)
        real_X = X.groupby('real').get_group(1).drop('real', axis=1)
        syn_X = X.groupby('real').get_group(0).drop('real', axis=1) if has_synthetic else pd.DataFrame()
        syn_y = y[syn_X.index] if has_synthetic else pd.DataFrame()

        scores = []
        for t, v in self.kf.split(real_X):
            X_train, X_val = real_X.iloc[t], real_X.iloc[v]
            y_train, y_val = y.iloc[t], y.iloc[v]   

            if has_synthetic:
                X_train = pd.concat((X_train, syn_X), ignore_index=True)
                y_train = pd.concat((y_train, syn_y), ignore_index=True)
            
            self.pipeline.fit(np.array(X_train).reshape(-1), y_train)
            y_pred = self.pipeline.predict(np.reshape(X_val, -1))
            
            acc = f1_score(y_val, y_pred)
            scores.append(acc)
            
        return scores

