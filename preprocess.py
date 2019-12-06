import numpy as np
import tensorflow as tf
import sklearn.neighbors as nn

gpu_available = tf.test.is_gpu_available()
print("GPU Available: ", gpu_available)


def get_train_data():
    # TODO: change to handle entire batch
    data = convert_to_LAB()
    l_images = data[:, 0]
    ab_images = data[:, 1:]
    labels = get_labels(ab_images)
    return l_images, ab_images, labels


def get_labels(ab_img):
    """
    Calculate labels to be passed into loss function
    :param ab_img: ab channels of images, shape (num_images, 2)
    :return: labels to pass into loss function
    """
    pass

def walk_data():
    """
    walk through data set directory
    :return: None, you should save all images to one directory
    """
    pass

def convert_to_LAB():
    """
    read images from the directory saved by walk_date, convert to LAB and store as npy
    :return: a numpy array
    """
    pass