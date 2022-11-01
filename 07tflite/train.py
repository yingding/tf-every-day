'''
This model trains a fashion mnist model, and convert to tensorflow lite mode with the following exposed functions
* train
* infer
* save
* restore
functions
'''
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from custom_model import Model, IMG_SIZE
from helper import create_default_checkpoint_subfolder

# print(f"Tensorflow verion: {tf.__version__}")
print(f"Tensorflow verion: {tf.version.VERSION}")

# quick version check for this code
assert tf.version.VERSION == "2.10.0"


fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

###
# Preprocess the data
# Pixel values in this dataset are between 0 and 255, and must be normalized to a value
# between 0 and 1 for processing by the model. Divide the values by 255 to make this adustment
###
train_images = (train_images / 255.0).astype(np.float32)
test_images = (test_images / 255.0).astype(np.float32)

## Convert the data labels to categorical values by performing one-hot encoding
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

NUM_EPOCHS = 100 #100
BATCH_SIZE = 100

epochs = np.arange(1, NUM_EPOCHS + 1, 1)
losses = np.zeros([NUM_EPOCHS])
m = Model()

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_ds = train_ds.batch(BATCH_SIZE)

for i in range(NUM_EPOCHS):
    for x, y in train_ds:
        result = m.train(x, y)

    losses[i] = result['loss'] 
    if (i + 1) % 10 == 0:
        print(f"Finished {i+1} epochs")
        print(f"   loss: {losses[i]:.3f}")

# Save the trained weights to a checkpoint
current_model_path = create_default_checkpoint_subfolder()
if current_model_path is not None:
    m.save(f"{current_model_path}/model.ckpt")

###
# Plot the training result
###

plt.plot(epochs, losses, label='Prfe-training')
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch')
plt.ylabel('Loss [Cross Entropy]')
plt.legend()

plt.show()