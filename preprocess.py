import numpy as np

l_images = np.load("data/gray_scale.npy")
ab_images = np.load("data/ab/ab1.npy")
ab_images = np.concatenate([ab_images, np.load("data/ab/ab2.npy")])
ab_images = np.concatenate([ab_images, np.load("data/ab/ab3.npy")])

def get_train_data():
    labels = get_labels(l_images, ab_images)
    return l_images, ab_images, labels

def get_labels():
    pass