from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import random

'''
First define a few utilities for preparing and visualizing your dataset.
'''
# switch to enable or disable show image
SHOW_IMAGES = False # True

def create_pairs(x: np.ndarray, digit_indices: list):
    """
    Positive and negative pair creation.
    Alternates between positive and negative pairs.
    @param: x, is the [?, 28, 28] shaped images collection
    @param: digit_indices, is a list with 10 elements,
            digit_indices[0] is an numpy.array of all images' indices with image label 0,
            digit_indices[n] is an numpy.array of all images' indices with image label n,
            n is element of [0-9], since there is only 10 label categories of keras mnist fashion image
    """
    pairs = []
    labels = []
    # stores the original fashion labels, so that the item can be identified later
    fashionLabels = []

    # get the smallest size of all 10 label categories and -1 so that i+1 will not index out of bound
    n = min([len(digit_indices[d]) for d in range(10)]) - 1

    for d in range(10): # 10 labels categories [0-9]
        for i in range(n): # from 0 to (smallest labeled category size - 1)
            # get the consecutive image pair from all the labels categories
            # only do this till the smallest labeled category is done.
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            # use array concatination to merge array
            # with an array of single element of similar image pair.
            pairs += [[x[z1], x[z2]]]
            fashionLabels += [[d, d]]
            # choose dn, which is randomly a different labeled category as the current one.
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            # construct a dissimilar image pair
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            # merge the dissimilar image pair to result pairs array
            pairs += [[x[z1], x[z2]]]
            fashionLabels += [[d, dn]]
            # in single loop, we added first a similar pair and then a dissimilar
            # thus, we added two labels to the labels array with the first 1 for similar pair of image
            # and second 0 for dissimilar pair image
            labels += [1, 0]


    return np.array(pairs), np.array(labels), np.array(fashionLabels)


def create_pairs_on_set(images: np.ndarray, labels: np.ndarray):
    """
    this function only creates double size of the smallest labeled category size.
    @param: images, numpy.ndarray with shape [n, 28, 28]
    @param: labels, numpy.ndarray with shape [n, 1]
    """
    # get the list of indices of images with [0-9] labels
    # the len(digit_indices) = 10,
    # digit_indices[0] returns an numpy.array of all images indices with label 0
    # Notice: np.where wrappes an numpy.array in an numpy.array, thus np.where(condition)[0] is used to unpack the outer array
    digit_indices = [np.where(labels == i)[0] for i in range(10)]
    pairs, y, fashion_label_pairs = create_pairs(images, digit_indices)
    # change the label type from uint8 to float32.
    y = y.astype('float32')
    return pairs, y, fashion_label_pairs


def show_image(image):
    if (SHOW_IMAGES):
        plt.figure()
        plt.imshow(image)
        plt.colorbar()
        plt.grid(False)
        plt.show()


def show_image_with_label(image: np.ndarray, label_id: int):
    if (SHOW_IMAGES):
        plt.figure()
        plt.imshow(image)
        plt.colorbar()
        plt.title(get_description_from_image_label(label_id))
        plt.grid(False)
        plt.show()


# create a global dict mapper
image2label: Dict = {
    0 : "T-shirt/top",
    1 : "Trouser",
    2 : "Pullover",
    3 : "Dress",
    4 : "Coat",
    5 : "Sandal",
    6 : "Shirt",
    7 : "Sneaker",
    8 : "Bag",
    9 : "Ankle boot"
}

def get_description_from_image_label(labelId : int) -> str :
    """
    get the label text by given the label id of a fashion image
    """
    return image2label.get(labelId, "Unknown")


def show_image_pairs(image_pair_array: np.ndarray, fashion_label_pair_array: np.ndarray,
                     image_similarity_array: np.array,
                     image_euclidean_dist_array: np.array,
                     pair_idx: int):

    # show images at this index
    show_image_with_label(image_pair_array[pair_idx][0], fashion_label_pair_array[pair_idx][0])
    show_image_with_label(image_pair_array[pair_idx][1], fashion_label_pair_array[pair_idx][1])

    # print the label for this pair
    print(f"Similarity Label: {image_similarity_array[pair_idx]}, (1 = similar; 0 = dissimilar)")
    print(f"Euclidean Dist. : {image_euclidean_dist_array[pair_idx]}, (0 ~= similar; 1 = dissimilar)")



def contrastive_loss_with_margin(margin):
    """
    contrastive loss with margin function
    :param margin:
    :return:
    """
    def contrastive_loss(y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
    return contrastive_loss


def compute_accuracy(y_true: np.ndarray , y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy with a fixed threshold on distances.
    @param: y_true, has shape (n, 1)
    @param: y_pred, has shape (n, 1), y_pred[0:1, :] = [[0.07228928]]
    """
    ## numpy.ravel == reshape(-1, order=order), returns a contiguous flattened array
    ## x = np.array([[1, 2, 3], [4, 5, 6]]), np.ravel(x) => array([1, 2, 3, 4, 5, 6])
    ## since the y_pred has shape (n, 1), used to ravel() to shape (n, ) as a 1-d array
    ## use numpy.array() < 0.5 to get and npmy.array of [true | false], indicating whether each index is smaller as 0.5
    ## Notice: As we are calculate the similarity of euclidean distance, the y_pred[n][0] < 0.5 mean the two images are similar
    ## Ultimately used np.mean(pred == y_true) to compare the true == 1 (1 for similar images) to get the percentage of right prediction.
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def plot_metrics(history: np.ndarray, metric_name: str, title:str, ylim=5):
    plt.title(title)
    plt.ylim(0,ylim)
    plt.plot(history.history[metric_name],color='blue',label=f"train: {metric_name}")
    plt.plot(history.history['val_' + metric_name], \
             color='green',label=f"validation: val_{metric_name}")
    plt.legend()
    plt.show()


# Matplotlib config
def visualize_images():
    plt.rc('image', cmap='gray_r')
    plt.rc('grid', linewidth=0)
    plt.rc('xtick', top=False, bottom=False, labelsize='large')
    plt.rc('ytick', left=False, right=False, labelsize='large')
    plt.rc('axes', facecolor='F8F8F8', titlesize="large", edgecolor='white')
    plt.rc('text', color='a8151a')
    plt.rc('figure', facecolor='F0F0F0')# Matplotlib fonts


# utility to display a row of digits with their predictions
def display_images(left: np.ndarray, right: np.ndarray, left_labels: list, right_labels: list,\
                   predictions: np.ndarray, labels: np.ndarray, title, n):
    """
    @param left, numpy.ndarray of images with shape (10, 28, 28)
    @param right, numpy.ndarray of images with shape (10, 28, 28)
    @param left_label, list of label text lenth 10
    @param right_label, list of label text lenth 10
    @param predictions, numpy.ndarray of predition with shape (10, ) is actually 1-d array
    @param labels, numpy.ndarray of predition with shape (10, ) is actually 1-d array
    """
    plt.figure(figsize=(17,3))
    plt.title(title)
    plt.yticks([])
    # plt.xticks([])
    # show the left label text
    plt.xticks([28*x+14 for x in range(n)], left_labels)
    plt.grid(None)
    # use swapaxes and reshap to concatenate the n image together.
    left = np.reshape(left, [n, 28, 28])
    left = np.swapaxes(left, 0, 1)
    left = np.reshape(left, [28, 28*n])
    plt.imshow(left)
    plt.figure(figsize=(17,3))
    plt.yticks([])

    # y is float y:.5f use f print to show float with 5 decimal
    right_compound_labels = map(lambda x, y: f"{x}\n{y:.5f}", right_labels, predictions)
    plt.xticks([28*x+14 for x in range(n)], right_compound_labels)
    # plt.xticks([28*x+14 for x in range(n)], predictions)


    for i,t in enumerate(plt.gca().xaxis.get_ticklabels()):
        if predictions[i] > 0.5: t.set_color('red') # bad predictions in red
    plt.grid(None)
    right = np.reshape(right, [n, 28, 28])
    right = np.swapaxes(right, 0, 1)
    right = np.reshape(right, [28, 28*n])
    plt.imshow(right)
    plt.show()


# calculate the left and right images label text
def label_unpack(label_pair_array: np.ndarray, indexes: np.ndarray) :
    label_pairs = label_pair_array[indexes]
    left_label_id, right_label_id = label_pairs[:, 0], label_pairs[:, 1]
    left_label = list(map(get_description_from_image_label, left_label_id))
    right_label = list(map(get_description_from_image_label, right_label_id))
    return left_label, right_label

