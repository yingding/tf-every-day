# About this repo

Following the instruction of On-device Training to 
1. train a mnist model (tf_train.py)
2. convert the trained tf model to tflite using compression (tflite_convert.py)
3. inference the compressed tflite model (tflite_inference.py)
4. retrain tflite model on python, or android device (tflite_retrain.py)
5. inference the retained tflite model from checkpoint (tflite_retrain_inference.py)


## Reference:
* Tuturial On-device training: https://www.tensorflow.org/lite/examples/on_device_training/overview#build_a_model_for_on-device_training
* Blog post of On-device training with TensorFlow Lite: https://blog.tensorflow.org/2021/11/on-device-training-in-tensorflow-lite.html
* Another MobileNetV2 example of On-device training on Android: https://github.com/tensorflow/examples/tree/master/lite/examples/model_personalization


# Future Works

* call bash operator, to execute the python scripts in airflow: https://stackoverflow.com/questions/41730297/python-script-scheduling-in-airflow/41731397#41731397
* Run multiple script as subprocesses: https://stackoverflow.com/questions/68980171/running-multiple-python-scripts-with-subprocess-python/68980201#68980201 


