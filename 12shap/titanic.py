from utils.datahelper import (
    KaggleData, 
    current_dir_subpath,
    profile, 
    feature_correlation, 
    na_columns,
    fill_missing_values_with_mean,
    DataVisualizer,
)

from utils.modelhelper import (
    ProbBinaryClassifier,
    ModelValidator,
    ModelExplainer,
    ModelKernelExplainer
)

from matplotlib import pyplot as plt
import seaborn as sns

import pandas as pd
from pandas import DataFrame

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


@ModelExplainer.valid_index
def get_details(df: DataFrame, idx: int, org_df: DataFrame) -> DataFrame:
    """get the passenger info from the original data frame based on the index position of validation set
    @param df: 
    @param idx: index position, this must be idx so that decorator works
    """
    # get the index name from the original raw dataset, from the index position of validation data set
    index_name = df.iloc[idx].name
    # use slicing on the same index name to get the passenger info as a DataFrame obj
    return org_df.loc[index_name: index_name]


def filter_values(x):
    """x is tuple with two positions"""
    match x[0]:
        case "Age":
            return (x[0], round(x[1]))
        case "Fare":
            return (x[0], round(x[1], 4)) # returns float with rounded 4 decimals
        case _:
            return x 


def explain_prediction(explainer: ModelExplainer, idx_pos: int, batch_df: DataFrame, org_df: DataFrame) -> None:
    # passager_df = get_details(df=X_valid, idx=idx_pos, org_df = train_X_raw_df)
    passager_df = get_details(df=batch_df, idx=idx_pos, org_df = org_df)
    print(passager_df)
    explainer.force_plot(idx = idx_pos)
    explainer.waterfall_plot(idx = idx_pos)


def main(dark_mode: bool):
    # create a visualization helper object
    visual_helper = DataVisualizer(dark_mode=dark_mode)

    titanic_train_path = current_dir_subpath("data/train.csv")
    titanic_test_path = current_dir_subpath("data/test.csv")
    label_name = "Survived"
    one_hot_cols = ["Sex"]

    titanic = KaggleData(
        train_path = titanic_train_path,
        test_path = titanic_test_path,
        label_col=label_name
    )

    # load all raw unprocessed data as DataFrame, label as Series
    # one_hot_cols transfers categorical column to one_hot encoded column 
    train_X_raw_df, test_X_raw_df, train_raw_y = titanic.load(one_hot_cols=one_hot_cols)
    all_X_raw_df = titanic.load_all(one_hot_cols=one_hot_cols)

    # detect the numerical features for building classifier
    num_features = all_X_raw_df.describe().columns.to_list()

    profile(train_X_raw_df, title="Profile of Raw Training Dataset")
    print("\n" + "#" * 20)
    profile(test_X_raw_df, title="Profile of Raw Test Dataset")
    
    titanic.boxplot_dist(
        ["Fare", "Age"], marker="x", orient="h", 
        legend="upper right", dark_mode=dark_mode)
    
    titanic.boxplot_dist(
        ['Pclass', 'SibSp', 'Parch'], marker="x", orient="h", 
        legend="upper right", dark_mode=dark_mode)
    

    """Date Inputation"""
    train_X_raw_df, train_mean_dict = fill_missing_values_with_mean(
        df=train_X_raw_df, pop_df=all_X_raw_df, filter_cols=num_features, 
        filter_func=filter_values)
    print(train_mean_dict)
    print(f"no. of numerical cols. has NaN values: {len(na_columns(train_X_raw_df, num_features))}")
    
    test_X_raw_df, test_mean_dict = fill_missing_values_with_mean(
    df=test_X_raw_df, pop_df=all_X_raw_df, filter_cols=num_features, 
    filter_func=filter_values)
    print(test_mean_dict)
    print(f"no. of numerical cols. has NaN values: {len(na_columns(test_X_raw_df, num_features))}")

    """Examining the correclation of numerical features"""
    train_X_y_raw_df = pd.concat([train_X_raw_df, train_raw_y], axis=1)
    threshold = 0.2
    corr_df, high_corr_df = feature_correlation(train_X_y_raw_df, label=label_name, threshold=threshold)
    print(corr_df)
    visual_helper.display_feature_correlation(corr_df=corr_df)
    # display the features with high correlation to the label
    print(high_corr_df)

    """select features"""
    # select only numerical features
    selected_features = num_features.copy()
    selected_features.remove("PassengerId")

    # split the training data
    X_train, X_valid, y_train, y_valid = titanic.split(
        titanic.select_cols(df=train_X_raw_df, cols=selected_features), 
        train_raw_y, test_size=0.2, random_state=10)
    
    
    """
    Training Xgboost model
    """
    param_grid = {
        'n_estimators': range(6, 10),
        'max_depth' : range(3, 8),
        'learning_rate' : [.2, .3, .4],
        'colsample_bytree' : [.7, .8, .9, 1]
    }
    
    xgb = XGBClassifier()
    # Searching for the best parameters
    g_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, n_jobs=1, verbose=0, return_train_score=True)

    # Fitting the model using best parameters found
    g_search.fit(X_train, y_train)

    # print the best parameters found
    print(g_search.best_params_)
    # valid 
    # g_search.score(X_valid, y_valid)
    model = g_search

    predicts = model.predict(X_valid)
    
    """
    Training Gender Based Naiv model
    """
    # feature_position = ProbBinaryClassifier.feature_position(X_train, "Sex_female")
    # # config model
    # model = ProbBinaryClassifier(feature_position, 1)
    # # Train model
    # model.fit(X_train, y_train)
    # # Validate model
    # predicts = model.predict(X_valid)

    """
    Validate model
    """
    validator = ModelValidator(y_valid, predicts, dark_mode=dark_mode)
    scores_dict = validator.evaluate()
    validator.print_eval_result(scores_dict)
    validator.display_confusion_matrix()
    validator.display_roc_curve()


    """
    Create model explaination
    """
    # Create Explainer callable for X_valid data set
    explainer = ModelExplainer(model=model, data=X_valid, dark_mode=dark_mode)
    # explainer = ModelKernelExplainer(model=model, train_data=X_train, inference_data=X_valid, dark_mode=dark_mode)
    
    explain_prediction(explainer=explainer, idx_pos=0, 
                       batch_df=X_valid, org_df=train_X_raw_df) 
    
    explain_prediction(explainer=explainer, idx_pos=100, 
                       batch_df=X_valid, org_df=train_X_raw_df)
    
    explainer.beeswarm_plot()

    explainer.summary_plot()

    
if __name__ == '__main__':
    dark_mode = True
    main(dark_mode=dark_mode)