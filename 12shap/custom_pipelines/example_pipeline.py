from zenml.pipelines import pipeline
from util import get_local_time_str
from baseline import baseline_trainer, load_data, shap_explainer
from xgboost.sklearn import XGBRegressor

@pipeline
def my_pipeline(step_1, step_2, step_3):
    X_train, y_train = step_1()
    base_regressor: XGBRegressor = step_2(X_train, y_train)
    step_3(X_train, base_regressor)

my_pipeline_instance = my_pipeline(
    step_1=load_data(),
    step_2=baseline_trainer(),
    step_3=shap_explainer()
)

# https://docs.zenml.io/starter-guide/pipelines
my_run_name=f"my_example_test_pipeline_run_{get_local_time_str(target_tz_str='Europe/Berlin')}" 
my_pipeline_instance.run(run_name=my_run_name, unlisted=True, enable_cache=False)
