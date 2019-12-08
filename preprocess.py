import numpy as np
import tensorflow as tf
from os import walk
from os.path import join
from shutil import copyfile
import cv2
import sklearn.neighbors as nn
from hyperparameters import IMAGE_HEIGHT, IMAGE_WIDTH, SIGMA

gpu_available = tf.test.is_gpu_available()
print("GPU Available: ", gpu_available)


def get_train_data():
    # TODO: change to handle entire batch
    data = convert_to_LAB()
    l_images = data[:, :, :, 0]
    ab_images = data[:, :, :, 1:]
    return l_images, ab_images, labels


def get_labels(ab_img):
    """
    Calculate labels to be passed into loss function
    :param ab_img: ab channels of images, shape (num_images, 2)
    :return: labels to pass into loss function
    """
    pass

def walk_data2():
    """
    walk through data set directory
    :return: None, you should save all images to one directory
    """
    all_files = []
    data_dir = "SUN2012/Images/"
    for root, subfolder, files in walk(data_dir):
        for file in files:
            if file.endswith('.jpg'):
                all_files.append(join(root, file))
                # copyfile(join(root, file), join("preprocessed/", file))

    return all_files


def convert_to_LAB():
    """
    read images from the directory saved by walk_date, convert to LAB and store as npy
    :return: a numpy array
    """
    jpegs = walk_data()
    lab = []
    for img in jpegs:
        im = cv2.imread(img)
        lab_im = cv2.cvtColor(im.astype('uint8'), cv2.COLOR_RGB2LAB)
        lab.append(lab_im)


    lab = np.asarray(lab)
    return lab

get_train_data()