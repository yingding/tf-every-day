import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils.helper import (
    create_default_tflite_model_path, 
    create_default_tflite_checkpoint_path
)
from utils.visual import (
    compare_logits,
    plot_images
)


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

another_interpreter = tf.lite.Interpreter(model_path=TFLITE_FILE_PATH)
another_interpreter.allocate_tensors()

# call the infer signatures of TensorFlow Lite
infer = another_interpreter.get_signature_runner("infer")
restore = another_interpreter.get_signature_runner("restore")

# infer returns dict object with 'logits' as key and an array of logits
logits_lite_before = infer(x=train_images[:1])['logits'][0]

print(logits_lite_before)
print(f"predicted: {np.argmax(logits_lite_before)}")
print(f"groundtruth: {train_labels[:1][0]}")

# Restore the trained weights from tflite model.ckpt, param checkpoint_path defined
# custom_model.py restore signature
tflite_checkpoint_path = create_default_tflite_checkpoint_path(check_point_name="model.ckpt")
if os.path.exists(tflite_checkpoint_path):
    # rettore the interpreter with a checkpoint
    restore(checkpoint_path=np.array(tflite_checkpoint_path, dtype=np.string_))

logits_lite_after = infer(x=train_images[:1])['logits'][0]

compare_logits({'Before': logits_lite_before, 'After': logits_lite_after}, plt_func=plt.show)


####
# Influence all the test images
####

infer = another_interpreter.get_signature_runner("infer")
# the return of infer signature is defined in the custom model with a dictionary output logits and output
# the output is a softmax of the logits, to maximize the probability
result = infer(x=test_images)

#print(result.keys())
#print(result["logits"])
#print(result["output"])

# argmax of the row vectors


predictions = np.argmax(result['output'], axis=1)
# test_labels = tf.keras.utils.to_categorical(test_labels)
# true_labels = np.argmax(test_labels, axis=1)

# the test_labels is a 1D array, without the transformation to logits
plot_images(test_images, 
    predictions=predictions, 
    true_labels=test_labels,
    title="tflite model performance, red: false prediction",
    plt_func=plt.show
)