'''
Build the model
'''

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda
from tensorflow.keras import backend as K
from typing import Dict
from tensorflow.python.keras.utils.vis_utils import plot_model

'''
define some utilities for building our model.
'''

def initialize_base_network():
    input = Input(shape=(28,28,), name="base_input")
    x = Flatten(name="flatten_input")(input)
    x = Dense(128, activation='relu', name="first_base_dense")(x)
    x = Dropout(0.1, name="first_dropout")(x)
    x = Dense(128, activation='relu', name="second_base_dense")(x)
    x = Dropout(0.1, name="second_dropout")(x)
    x = Dense(128, activation='relu', name="third_base_dense")(x)

    return Model(inputs=input, outputs=x)


def euclidean_distance(vects):
    """
    uses K.epsilon so that the distance is not zero if the vects are identical,
    to prevent backprop of zero.
    """
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    # while merging the two euclidean distance vector, the merged output will be the same as the either input.
    # either the vector size of shape1 or shape2 will be fine.
    # n = shape1[0] = shape2[0] is the batch size, 1 is the value of euclidean distance float value.
    shape1, shape2 = shapes
    return (shape1[0], 1)

class SiameseANN():
    def __init__(self):
        self.base_network = initialize_base_network()

        # create the left input and point to the base network
        self.input_a = Input(shape=(28,28,), name="left_input")
        self.vect_output_a = self.base_network(self.input_a)

        # create the right input and point to the base network
        self.input_b = Input(shape=(28,28,), name="right_input")
        self.vect_output_b = self.base_network(self.input_b)

        # measure the similarity of the two vector outputs
        self.output = Lambda(euclidean_distance, name="output_layer", \
                             output_shape=eucl_dist_output_shape)([self.vect_output_a, self.vect_output_b])

        # specify the inputs and output of the model
        self.model = Model([self.input_a, self.input_b], self.output)

    def plot_base_network(self):
        plot_model(self.base_network, show_shapes=True, show_layer_names=True, to_file='base-model.png')

    def summary_base_network(self):
        # examing the number of weights among the layers, which is param number
        # (784 + 1) * 128 = 100480 , +1 is bias unit for every neuron unit of 128 in first_base_dense layer.
        # (128 + 1) * 128 = 16512 , for the weights from dropout layer to dense layer
        self.base_network.summary()

    def getModel(self):
        return self.model

    def plot_model(self):
        # plot model graph
        plot_model(self.model, show_shapes=True, show_layer_names=True, to_file='outer-model.png')

    def summary_model(self):
        self.model.summary()
