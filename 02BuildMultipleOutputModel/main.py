"""
This example shows how you can build models with more than one output.
The dataset we will be working on is available from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Energy+efficiency).
It is an Energy Efficiency dataset which uses the bulding features
(e.g. wall area, roof area) as inputs and has two outputs: Cooling Load and Heating Load.
Let's see how we can build a model to train on this data.

Attribution: this example is based on DeepLearn.AI Coursera Course - TensorFlow: Advanced Techniques

@author: Yingding Wang
"""

# try:
#     # %tensorflow_version only exists in Colab.
#     %tensorflow_version 2.x
# except Exception:
#     pass

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# pip3 install xlrd, openpyxl
import pandas as pd
from pandas.core.frame import DataFrame
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.vis_utils import plot_model

'''
## Utilities: 

We define a few utilities for data conversion and visualization to make our code more neat.
'''

def format_output(data: DataFrame):
    """
    returns the y1, y2 in tuple
    """
    # pop remove the series from the original DataFrame
    y1 = data.pop('Y1')
    y1 = np.array(y1)
    y2 = data.pop('Y2')
    y2 = np.array(y2)
    return y1, y2


def norm(x: DataFrame, train_stats: DataFrame):
    """
    Z-score Normalization, Standardization, feature scaling to standard normal N(0,1)
    :param x:
    :return:
    """
    # x - train_states['mean'], equivalent to x.substract(train_states['mean'], axis='columns')
    return (x - train_stats['mean']) / train_stats['std']


def plot_diff(y_true, y_pred, title=''):
    plt.scatter(y_true, y_pred)
    plt.title(title)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    plt.plot([-100, 100], [-100, 100])
    plt.show()


def plot_metrics(history, metric_name, title, ylim=5):
    """
    plot the training loss and the validation loss
    """
    plt.title(title)
    plt.ylim(0, ylim)
    plt.plot(history.history[metric_name], color='blue', label=f'train loss: {metric_name}')
    plt.plot(history.history['val_' + metric_name], color='green', label=f'validation loss: {metric_name}')
    plt.legend()
    # use explicit name to override the plt.plot(..., label="")
    # plt.legend({f'validation loss: {metric_name}', f'train loss: {metric_name}'})
    # plt.legend({'blue','green'},'Location','northest')
    plt.show()


'''
## Prepare the Data

download the dataset and format it for training.
'''
# Get the data from UCI dataset
URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx'

# Use pandas excel reader
df = pd.read_excel(URL)


# This function: df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
# (https://stackoverflow.com/questions/43983622/remove-unnamed-columns-in-pandas-dataframe/43983654#43983654)
#
# use tilde unary operator to bitwise inverse the bool array
# (https://stackoverflow.com/questions/13600988/python-tilde-unary-operator-as-negation-numpy-bool-array)
# where df.columns.str.contains('^Unnamed') returns numpy.array with all columns name begins with Unnamed.
#
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]


# use df.sample(frac=1) to return the whole fraction of all samples in a random order
# reset_index(drop=true), reset_index in order and do not insert the older index into dataframe.
# with df.sample(frac=1).reset_index(drop=True) to get a random resorted dataset.
df = df.sample(frac=1).reset_index(drop=True)

# Split the data into train and test with 80 train / 20 test
# sklearn.model_selection
train, test = train_test_split(df, test_size=0.2)

# Generate the descriptive statistics of a DataFrame.
train_stats = train.describe()
# remove the "Y1" and "Y2" from the statistics DF and reverse the column and row
train_stats.pop('Y1')
train_stats.pop('Y2')
train_stats = train_stats.transpose()

# Get Y1 and Y2 as the 2 outputs and format them as np arrays
train_Y = format_output(train)
test_Y = format_output(test)

# Scaling the training and test data using the training mean
norm_train_X = norm(train, train_stats)
norm_test_X = norm(test, train_stats)


'''
## Build the Model

Here is how we'll build the model using the functional syntax. Notice that we can specify a list of outputs (i.e. `[y1_output, y2_output]`) when we instantiate the `Model()` class.
'''

# Define model layers.
input_layer = Input(shape=(len(train .columns),))
first_dense = Dense(units='128', activation='relu')(input_layer)
second_dense = Dense(units='128', activation='relu')(first_dense)

# Y1 output will be fed directly from the second dense
y1_output = Dense(units='1', name='y1_output_')(second_dense)
third_dense = Dense(units='64', activation='relu')(second_dense)

# Y2 output will come via the third dense
y2_output = Dense(units='1', name='y2_output_')(third_dense)

# Define the model with the input layer and a list of output layers
model = Model(inputs=input_layer, outputs=[y1_output, y2_output])

# Plot model graph
plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')
print(model.summary())


'''
## Configure parameters

specify the optimizer as well as the loss and metrics for each output.
'''

# Specify the optimizer, and compile the model with loss functions for both outputs
optimizer = tf.keras.optimizers.SGD(lr=0.001)
model.compile(optimizer=optimizer,
              loss={'y1_output_': 'mse', 'y2_output_': 'mse'}, # the keys are the name of the layer/function.
              metrics={'y1_output_': tf.keras.metrics.RootMeanSquaredError(),
                       'y2_output_': tf.keras.metrics.RootMeanSquaredError()})


'''
## Train the Model

[More](https://www.tensorflow.org/guide/keras/train_and_evaluate) to train and evaluate with kera build-in methods.
'''

# With 500 Epochs, model might have better accuracy, but it takes some time to train.
EPOCHS = 200 # 500
BATCH_SIZE = 10
# Train the model for 500 epochs
history = model.fit(norm_train_X, train_Y,
                    epochs=EPOCHS, batch_size=BATCH_SIZE,
                    # We pass some validation for
                    # monitoring validation loss and metrics
                    # at the end of each epoch
                    validation_data=(norm_test_X, test_Y))

'''
## Evaluate the Model and Plot Metrics
'''

# Test the model and print loss and mse for both outputs
loss, Y1_loss, Y2_loss, Y1_rmse, Y2_rmse = model.evaluate(x=norm_test_X, y=test_Y)
print("Loss = {}, Y1_loss = {}, Y1_mse = {}, Y2_loss = {}, Y2_mse = {}".format(loss, Y1_loss, Y1_rmse, Y2_loss, Y2_rmse))


# Plot the loss and mse
Y_pred = model.predict(norm_test_X)
plot_diff(test_Y[0], Y_pred[0], title='Y1')
plot_diff(test_Y[1], Y_pred[1], title='Y2')
plot_metrics(history, metric_name='y1_output__root_mean_squared_error', title='Y1 RMSE', ylim=6)
plot_metrics(history, metric_name='y2_output__root_mean_squared_error', title='Y2 RMSE', ylim=7)
