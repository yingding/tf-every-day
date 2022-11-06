import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils.helper import (
    create_default_tflite_model_path, 
    create_default_data_exchange_path,
    create_default_tflite_checkpoint_path,
    PreviousTrainingResults
)
from utils.visual import (
    display_training_loss,
    PlotLineData
)

"""
This retrain module simulate the tflite model training on android device with existing training data
"""

## Load the data
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

###
# Preprocess the data
# Pixel values in this dataset are between 0 and 255, and must be normalized to a value
# between 0 and 1 for processing by the model. Divide the values by 255 to make this adustment
###
train_images = (train_images / 255.0).astype(np.float32)

## Convert the data labels to categorical values by performing one-hot encoding
train_labels = tf.keras.utils.to_categorical(train_labels)


# https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_python
TFLITE_FILE_PATH = create_default_tflite_model_path("model.tflite")
interpreter = tf.lite.Interpreter(model_path=TFLITE_FILE_PATH)
interpreter.allocate_tensors()

# get the training method signature
train = interpreter.get_signature_runner("train")

NUM_EPOCHS = 50
BATCH_SIZE = 100
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_ds = train_ds.batch(BATCH_SIZE)

# previous epochs
data_exchange_path = create_default_data_exchange_path(exchange_file_name="previous_training_results_dataclass")
previous_traing_results = PreviousTrainingResults.load(data_exchange_path)
epochs = previous_traing_results.epochs
losses = previous_traing_results.losses

more_epochs = np.arange(epochs[-1] + 1, epochs[-1] + NUM_EPOCHS + 1, 1)
more_losses = np.zeros([NUM_EPOCHS])

####
# Retrain a tflite flatt buffer model
####
for i in range(NUM_EPOCHS):
    for x, y in train_ds:
        result = train(x=x, y=y)
        more_losses[i] = result['loss']
    if (i + 1) % 10 == 0:
        print(f"Finished {i+1} epochs")
        print(f"   loss: {more_losses[i]:.3f}")

####
# save retrained tflite model as tflite checkpoint
####
tflite_checkpoint_path = create_default_tflite_checkpoint_path("model.ckpt") 
## save the checkpoint
save = interpreter.get_signature_runner("save")
# checkpoint_path is defined in the custom_model.py , save mothod
save(checkpoint_path=np.array(tflite_checkpoint_path, dtype=np.string_))


# show the retain result, the loss of On device continous
display_training_loss(lines=[
        PlotLineData(x_values=epochs, y_values=losses, label="Pre-training"),
        PlotLineData(x_values=more_epochs, y_values=more_losses, label="On device"),
    ],
    plt_func=plt.show
)