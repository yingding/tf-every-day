"""
This example will go trough creating and training a multi-input model. You will build a basic Siamese Network to find the similarity or dissimilarity between items of clothing. You can either train with the fliped similarity label and contrastive loss or train with the euclidean distance label and mse loss, both way will achieve the same result.

Attribution: this example is based on DeepLearn.AI Coursera Course - TensorFlow: Advanced Techniques

@author: Yingding Wang
"""

# try:
#     # %tensorflow_version only exists in Colab.
#     %tensorflow_version 2.x
# except Exception:
#     pass

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.keras import backend as K
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import random

from utils import *;
from model import SiameseANN;

'''
## Prepare the Dataset
You can now download and prepare our train and test sets. You will also create pairs of images that will go into the multi-input model.
'''

# load the keras fashion mnist dataset
'''
  This is a dataset of 60,000 28x28 grayscale images of 10 fashion categories,
  along with a test set of 10,000 images. This dataset can be used as
  a drop-in replacement for MNIST. The class labels are:

  | Label | Description |
  |:-----:|-------------|
  |   0   | T-shirt/top |
  |   1   | Trouser     |
  |   2   | Pullover    |
  |   3   | Dress       |
  |   4   | Coat        |
  |   5   | Sandal      |
  |   6   | Shirt       |
  |   7   | Sneaker     |
  |   8   | Bag         |
  |   9   | Ankle boot  |

   Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

      **x_train, x_test**: uint8 arrays of grayscale image data with shape
        (num_samples, 28, 28).

      **y_train, y_test**: uint8 arrays of labels (integers in range 0-9)
        with shape (num_samples,). 
'''
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Examing the data structure
first_train_image: np.array = train_images[0, :, :]
print(f"train_images.shape : {train_images.shape}")
print(f"train_labels.shape : {train_labels.shape}")
print(f"train_labels.type : {type(train_labels)}")
# print(f"first train_image: {first_train_image}")
first_train_image_label = train_labels[0]

# show the first train_image in grey scale
show_image_with_label(first_train_image, first_train_image_label)

'''prepare train and test sets'''
# since the 28x28 fashion mnist image type is uint8, it need to casted to float32
# so that float value can be normalized later
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

# normalize values
train_images = train_images / 255.0
test_images = test_images / 255.0

# show first image with normalized pixel values
show_image_with_label(train_images[0,:,:], first_train_image_label)


# create pairs on train and test sets
# the xx_fashion_label_pairs contains the original label pair for the training or test images pairs.
tr_pairs, tr_y, tr_fashion_label_pairs = create_pairs_on_set(train_images, train_labels)
ts_pairs, ts_y, ts_fashion_label_pairs = create_pairs_on_set(test_images, test_labels)

# flip the tr_y, ts_y label for euclidean distance, since where the label similarity is 1, the euclidean distance shall be 0 and vice versa
tr_y_euc = 1 - tr_y
ts_y_euc = 1 - ts_y

'''
You can see a sample pair of images below.
'''
# array index
this_pair = 8
show_image_pairs(ts_pairs, ts_fashion_label_pairs, ts_y, ts_y_euc, this_pair)

# Examine other image pairs: the first two training pairs
show_image_pairs(tr_pairs, tr_fashion_label_pairs, tr_y, tr_y_euc, 0)
show_image_pairs(tr_pairs, tr_fashion_label_pairs, tr_y, tr_y_euc, 1)


'''
Build the Model
'''

model_container = SiameseANN()
# show the structure of the stem of multiple inputs network
model_container.plot_base_network()
model_container.summary_base_network()

# show the whole structure of the siamese ANN with two inputs
model_container.plot_model()
model_container.summary_model()

# assign the custom keras model structure with two image inputs
model: Model = model_container.getModel()

'''
## Train the Model
define the custom loss for our network and start training.
'''
rms = RMSprop()
EPOCHS = 5 # 20
MARGIN = 1 # 1

'''
Notice: the loss function is only involved in the training,
        in inferencing the network will construct forwards calculation
        to computed the euclidean simularity.
'''

# model.compile(loss=contrastive_loss_with_margin(margin=MARGIN), optimizer=rms)
# history = model.fit([tr_pairs[:,0], tr_pairs[:,1]], tr_y, epochs=EPOCHS, batch_size=128,\
#                     validation_data=([ts_pairs[:,0], ts_pairs[:,1]], ts_y))

'''
Uncomment the following line to train with 'mse' loss and 'euclidean distance labels'

use the euclidean label instead of the flipped similarity label: tr_y_euc = 1 - tr_y and ts_y_euc = 1 - ts_y
'''

model.compile(loss='mse', optimizer=rms)
history = model.fit([tr_pairs[:,0], tr_pairs[:,1]], tr_y_euc, epochs=EPOCHS, batch_size=128,\
                    validation_data=([ts_pairs[:,0], ts_pairs[:,1]], ts_y_euc))

'''
## Model Evaluation

evaluate the model by computing the accuracy and observing the metrics during training.
'''
loss: float = model.evaluate(x=[ts_pairs[:,0],ts_pairs[:,1]], y=ts_y)

# the shape of y_pred_train is (n, 1) , y_pred[0:1, :] = [[0.07228928]], each prediction is wrapped with an array.
y_pred_train: np.ndarray = model.predict([tr_pairs[:,0], tr_pairs[:,1]])

# used tr_y similarity label here, no the euclidean label
train_accuracy = compute_accuracy(tr_y, y_pred_train)

y_pred_test = model.predict([ts_pairs[:,0], ts_pairs[:,1]])

test_accuracy = compute_accuracy(ts_y, y_pred_test)

print("Loss = {}, Train Accuracy = {} Test Accuracy = {}".format(loss, train_accuracy, test_accuracy))


# we haven't give a name to the contrastive_loss_with_margin 1 so the metric name is just loss
# as we can see the validation loss goes slightly up arround 5 epochs of total 20 epoch.
# To mitigate the overfitting, we may want to early stop arround 5th epoch.
plot_metrics(history=history, metric_name='loss', title=f"contrastive loss with margin (margin = {MARGIN})", ylim=0.2)

'''
see sample results for 10 pairs of items below.
'''
# np.squeeze is very similar to np.raven to unpack the wrapped array to make a flatten array.
y_pred_train = np.squeeze(y_pred_train)
# pick 10 random idx
num_images = 10
indexes = np.random.choice(len(y_pred_train), size=num_images)
left_labels, right_labels = label_unpack(tr_fashion_label_pairs, indexes)


display_images(tr_pairs[:, 0][indexes], tr_pairs[:, 1][indexes], left_labels, right_labels, \
               y_pred_train[indexes], tr_y[indexes], \
               "clothes and their dissimilarity with euclidean distance", num_images)















