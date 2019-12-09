import numpy as np
import tensorflow as tf
from os import walk
from os.path import join
from shutil import copyfile
from skimage.color import lab2rgb, rgb2lab
from skimage.io import imread

from hyperparameters import IMAGE_HEIGHT, IMAGE_WIDTH

gpu_available = tf.test.is_gpu_available()
print("GPU Available: ", gpu_available)


def get_train_data():
    # TODO: change to handle entire batch
    data = convert_to_LAB()
    l_images = data[:, :, :, 0].reshape((-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1)).astype('float32')
    ab_images = data[:, :, :, 1:].astype('float32')
    return l_images, ab_images


def walk_data():
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
    for i, img in enumerate(jpegs):
        im = imread(img)
        lab_im = rgb2lab(im.astype('uint8'))
        lab.append(lab_im)

    lab = np.asarray(lab)
    return lab
