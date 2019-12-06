import numpy as np
import tensorflow as tf
import sklearn.neighbors as nn

l_images = np.load("data/gray_scale.npy")
ab_images = np.load("data/ab/ab1.npy")
ab_images = np.concatenate([ab_images, np.load("data/ab/ab2.npy")])
ab_images = np.concatenate([ab_images, np.load("data/ab/ab3.npy")])

gpu_available = tf.test.is_gpu_available()
print("GPU Available: ", gpu_available)


def get_train_data():
    # TODO: change to handle entire batch
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