from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

class ModelEvaluator:

    model: BaseEstimator
    pipeline: Pipeline

    def __init__(self, model: BaseEstimator, X_test, y_test) -> None:
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer()),
            ('classifier', model)    
        ], verbose=True)

    def evaluate(self, X, y):
        self.pipeline.fit(X, y)
        y_pred = self.pipeline.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)        
        return y_pred, accuracy
