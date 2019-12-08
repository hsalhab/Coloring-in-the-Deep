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
    labels = get_labels(ab_images)
    return l_images, ab_images, labels


def get_labels(ab_img):
    """
    Calculate labels to be passed into loss function
    :param ab_img: ab channels of images, shape (num_images, 2)
    :return: labels to pass into loss function
    """
    labels = None
    bin_centers = np.load("bin_centers.npy")
    knn = nn.NearestNeighbors(
        n_neighbors=5, algorithm='ball_tree').fit(bin_centers)
    pixel_idx = tf.constant(np.arange(0, ab_img.shape[1] * ab_img.shape[2])[:, np.newaxis], dtype=tf.int64)
    for i, img in enumerate(ab_img):
        print("img #{} our of {}".format(i, ab_img.shape[0]))
        label = np.zeros((IMAGE_HEIGHT * IMAGE_WIDTH, 313))
        ii = np.reshape(img, (-1, 2))
        print(ii.shape)
        distances, indices = knn.kneighbors(ii, 5)
        weights = np.exp(-distances ** 2 / (2 * SIGMA ** 2))
        weights = weights / np.sum(weights, axis=1)[:, np.newaxis]
        pixel_idx = np.arange(IMAGE_HEIGHT * IMAGE_WIDTH)[:, np.newaxis]
        label[pixel_idx, indices] = weights
        label = np.reshape(label, (1, IMAGE_WIDTH, IMAGE_HEIGHT, 313))
        if labels is None:
            labels = label
        else:
            labels = np.concatenate([labels, label])

    print(labels.shape)
    np.save("labels.npy", labels)
    return labels

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