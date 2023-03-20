"""
ClassifierMixin is a general type of Classification mode
"""

from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
from pandas import DataFrame
import random
import numpy as np


class ProbBinaryClassifier(BaseEstimator, ClassifierMixin):
    """
    this Classifier learns a Maximal Likelihood as Probability based on single feature associated with positive label,
    this probability is used to predict for a binary classifier 
    """
    @staticmethod
    def _decision(probabilty: float) -> int:
        """
        0, 1 decision making based on probability
        decorated with staticmethod so that it can also be called with self._decision 
        without self being passed as the first param implicity
        
        Reference:
        https://stackoverflow.com/questions/43587044/do-we-really-need-staticmethod-decorator-in-python-to-declare-static-method/43587154#43587154
        """
        return int(random.random() <= probabilty)
    
    @staticmethod
    def feature_position(df: DataFrame, col_name):
        """
        returns the index position of a given column name for a DataFrame.
        This is a helper method used for init a model 
        """
        return df.columns.get_loc(col_name)
    

    def __init__(self, feature_position, feature_value):
        # position in the training data for a categorical feature
        self.feature_position = feature_position
        # the true positive values of the categorical feature
        self.feature_value = feature_value


    def fit(self, X, y=None):
        # print(type(X))
        # print(type(y))
        X_y_df = pd.concat([X, y], axis=1)
        # get the probability of positive
        feature_positive_label = X_y_df.loc[X_y_df.iloc[:, self.feature_position] == self.feature_value].iloc[:, -1]
        # get the random rate
        self._rate = sum(feature_positive_label) / len(feature_positive_label)
        # round two decimal digists
        self._rate = round(self._rate, 2)


    def predict(self, X):
        # both numpy.ndarray and pandas.DataFrame has .shape property
        # get 0 or 1 by probability, the dim 0 size is the same as input
        return np.array([self._decision(self._rate) for _ in range(0, X.shape[0])])


    def predict_proba(self, X, y=None):
        pass