import numpy as np
import tensorflow as tf
from os import walk
from os.path import join
from shutil import copyfile
from hyperparameters import BATCH_SIZE
from random import shuffle
import cv2

from hyperparameters import IMAGE_HEIGHT, IMAGE_WIDTH

gpu_available = tf.test.is_gpu_available()
print("GPU Available: ", gpu_available)

data_paths = []


def get_batch(batch_idx):
    batch = np.asarray([img2lab(file) for file in data_paths[batch_idx * BATCH_SIZE: (batch_idx + 1) * BATCH_SIZE]])
    return (batch[:, :, :, 0].reshape((-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1)).astype('float32'),
            batch[:, :, :, 1:].astype('float32'))


def fetch_data():
    """
    walk through data set directory
    :return: None, you should save all images to one directory
    """

    global data_paths
    data_dir = "SUN2012/Images/"
    for root, subfolder, files in walk(data_dir):
        for file in files:
            if file.endswith('.jpg'):
                data_paths.append(join(root, file))
                if len(data_paths) == 30:
                    return 30
                # copyfile(join(root, file), join("preprocessed/", file))

    return len(data_paths)


def shuffle_data():
    shuffle(data_paths)


def img2lab(file_name):
    img = cv2.imread(file_name)
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return ((lab_img / [100, 1, 1]) * [255, 1, 1]) - [0, 128, 128]
