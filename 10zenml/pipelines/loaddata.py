import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from zenml.steps import Output, step

@step
def digits_data_loader() -> Output(
    X_train=np.ndarray, X_test=np.ndarray, y_train=np.ndarray, y_test =np.ndarray
):
    """Loads the digits dataset as a tuple o flattened numpy arrays."""
    digits = load_digits()
    data = digits.images.reshape((len(digits.images), -1))
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.2, shuffle=False
    )
    return X_train, X_test, y_train, y_test
