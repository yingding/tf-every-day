### svc support vector classifier
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.svm import NuSVC

from zenml.steps import step

@step
def svc_trainer(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> ClassifierMixin: 
    """Train a sklearn NuSVC classfier."""
    model = NuSVC(gamma=0.001)
    model.fit(X_train, y_train)
    return model