"""
ClassifierMixin is a general type of Classification mode
"""

from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
from pandas import DataFrame, Series
import random
import numpy as np
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    f1_score, 
    roc_auc_score, # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    auc,
    roc_curve, 
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)


from numpy import ndarray
from typing import Tuple

class ModelValidator:
    """Binary Classification Model Validator"""
    def __init__(self, y_truth: Series, y_pred: ndarray, label_name_map: dict={0:"Perished", 1: "Survived"}, pos_label=1):
        self.y_truth = y_truth
        self.y_pred = y_pred
        self.label_name_map = label_name_map
        self.unknown_label_name = "unknown"
        # value list of categorical labels 
        self.cat_int_labels = self.y_truth.value_counts().index.to_list()
        # name list of categorical labels
        self.cat_char_labels = [ self.label_name_map.get(idx, self.unknown_label_name) for idx in self.cat_int_labels ]
        # positive label value for binary classification
        self.pos_label = pos_label

    def confusion_matrix(self) -> Tuple[DataFrame, ndarray]:
        """
        returns confusion matrix as a DataFrame
        with Columns and Row index of the Categorical labels
        """
        self.conf_mat = confusion_matrix(self.y_truth, self.y_pred, labels=self.cat_int_labels)
        # use both cat_char_labels as index and column names
        self.conf_mat_df = pd.DataFrame(self.conf_mat, index=self.cat_char_labels, columns=self.cat_char_labels)
        return self.conf_mat_df, self.conf_mat
    

    def accuracy_score(self):
        self.acc = accuracy_score(self.y_truth, self.y_pred)
        return self.acc
    

    def f1_score(self):
        self.f1 = f1_score(self.y_truth, self.y_pred)
        return self.f1
    
    
    def roc_curve(self)-> Tuple[ndarray, ndarray, ndarray, float]:
        """
        ROC: Receiver operating characteristic
        positive label is 1 is the survived label
        
        Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
        https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_display_object_visualization.html#sphx-glr-auto-examples-miscellaneous-plot-display-object-visualization-py
        
        AUC: The Area Under the Curve (AUC) is the measure of the ability of a binary classifier to distinguish between classes
        and is used as a summary of the ROC curve.
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
        
        """
        # fpr, tpr, thresholds = roc_curve(self.y_truth, self.y_pred, pos_label=)
        self.roc_fpr, self.roc_tpr, self.roc_thresholds = roc_curve(self.y_truth, self.y_pred, pos_label=self.pos_label)
        self.roc_auc = auc(self.roc_fpr, self.roc_tpr)
        return self.roc_fpr, self.roc_tpr, self.roc_thresholds, self.roc_auc
    
    def evaluate(self) -> dict:
        self.confusion_matrix()
        self.f1_score()
        self.accuracy_score()
        self.roc_curve()

        return {
            "conf_mat_df": self.conf_mat_df,
            "conf_mat": self.conf_mat,
            "f1": self.f1,
            "acc": self.acc,
            "roc_fpr": self.roc_fpr,
            "roc_tpr": self.roc_tpr,
            "roc_thresholds": self.roc_thresholds,
            "auc": self.roc_auc 
        }
    

    @staticmethod
    def print_eval_result(result: dict)-> None:
        """helper method to print the eval result"""
        for key, value in result.items():
            print(f"## {key}:")
            print(f"{value}\n")


    def display_roc_curve(self):
        """
        https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_display_object_visualization.html#sphx-glr-auto-examples-miscellaneous-plot-display-object-visualization-py
        """
        roc_display = RocCurveDisplay(
            fpr=self.roc_fpr, tpr=self.roc_tpr,roc_auc=self.roc_auc, 
            pos_label=self.pos_label).plot()    
        return roc_display
    

    def display_confusion_matrix(self):
        cm_display = ConfusionMatrixDisplay(self.conf_mat, display_labels=self.cat_char_labels).plot()
        return cm_display  



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
    

    def __init__(self, feature_position, pos_label):
        # position in the training data for a categorical feature
        self.feature_position = feature_position
        # the true positive values of the categorical feature
        self.feature_value = pos_label


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