'''
This testing module tests some basic function of the venv
'''
import tensorflow as tf
from utils.helper import (
    create_default_data_exchange_path, 
    SerializableList,
    PreviousTrainingResults,
)

print(f"Tensorflow verion: {tf.__version__}")

logits_path = create_default_data_exchange_path('logits')
print(f"{logits_path}")

a = SerializableList([1, 2])
print(a)

a.dump(logits_path)
b = SerializableList.load(logits_path)
print(b)

assert a == b

results_path = create_default_data_exchange_path('results')
pre_results = PreviousTrainingResults([1, 2], [3, 3])

pre_results.dump(results_path)
my_results = PreviousTrainingResults.load(results_path)

assert pre_results == my_results

print(pre_results)
print(my_results)