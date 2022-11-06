import tensorflow as tf
import numpy as np
from utils.helper import create_default_tflite_model_path

## Load the data
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

###
# Preprocess the data
# Pixel values in this dataset are between 0 and 255, and must be normalized to a value
# between 0 and 1 for processing by the model. Divide the values by 255 to make this adustment
###
train_images = (train_images / 255.0).astype(np.float32)
test_images = (test_images / 255.0).astype(np.float32)

# https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_python
TFLITE_FILE_PATH = create_default_tflite_model_path("model.tflite")
## use model_content to pass the tflite_model binary
## interpreter = tf.lite.Interpreter(model_content==tflite_model)

interpreter = tf.lite.Interpreter(model_path=TFLITE_FILE_PATH)
interpreter.allocate_tensors()

# call the infer signatures of TensorFlow Lite
infer = interpreter.get_signature_runner("infer")

# infer returns dict object with 'logits' as key and an array of logits
logits_lite = infer(x=train_images[:1])['logits'][0]

print(logits_lite)
print(f"predicted: {np.argmax(logits_lite)}")
print(f"groundtruth: {train_labels[:1][0]}")


