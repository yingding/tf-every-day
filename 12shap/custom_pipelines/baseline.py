import numpy as np
from zenml.steps import Output, step
from sklearn.datasets import load_diabetes
from pandas import DataFrame, Series
import xgboost
from xgboost.sklearn import XGBRegressor

@step
def load_data() -> Output(
    X_train=DataFrame, 
    y_train=Series
):
    """Load boston hausing data"""
    X_train, y_train = load_diabetes(return_X_y=True, as_frame=True)
    # print(type(X_train))
    # (442, 10)
    # print(type(y_train))
    # (442,)
    return X_train, y_train

@step
def baseline_trainer(
    X_train: DataFrame,
    y_train: Series
) -> XGBRegressor:
    # print(f"X_train type {type(X_train)}")
    # print(X_train.shape)
    # print(f"y_train type {type(y_train)}")
    # print(y_train.shape) 
    print(X_train.head())
    print(y_train.head())
    model = xgboost.XGBRegressor().fit(X_train, y_train)
    return model


@step 
def shap_explainer(X_train: DataFrame, model: XGBRegressor) -> None:
    import shap
    explainer = shap.Explainer(model)
    shap_values = explainer(X_train)
    # shap.plots.waterfall(shap_values[0])
    # shap.plots.force(shap_values)
    print(type(shap_values))
    print(shap_values[0, "bmi"])
    shap.plots.scatter(shap_values[:,"bmi"], color=shap_values)

