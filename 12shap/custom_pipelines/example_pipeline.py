from zenml.pipelines import pipeline
from util import get_local_time_str
from baseline import baseline_trainer

@pipeline
def my_pipeline(step_1):
    step_1()

my_pipeline_instance = my_pipeline(
    step_1=baseline_trainer(),
)

# https://docs.zenml.io/starter-guide/pipelines
my_run_name=f"my_example_test_pipeline_run_{get_local_time_str(target_tz_str='Europe/Berlin')}" 
my_pipeline_instance.run(run_name=my_run_name, unlisted=True, enable_cache=False)
