from zenml.pipelines import pipeline

from loaddata import digits_data_loader
from svc_estimator import svc_trainer
from util import get_local_time_str
from tf_estimator import tf_gpu_trainer

@pipeline
def first_pipeline(step_1, step_2, step_3):
    X_train, X_test, y_train, y_test = step_1()
    step_2(X_train, y_train)
    step_3(X_train, y_train, X_test, y_test)

first_pipeline_instance = first_pipeline(
    step_1=digits_data_loader(),
    step_2=svc_trainer(),
    step_3=tf_gpu_trainer()
)

# https://docs.zenml.io/starter-guide/pipelines
my_run_name=f"my_simple_test_run_{get_local_time_str(target_tz_str='Europe/Berlin')}" 
first_pipeline_instance.run(run_name=my_run_name, unlisted=True, enable_cache=False)