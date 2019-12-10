import sklearn.neighbors as nn
from hyperparameters import IMAGE_HEIGHT, IMAGE_WIDTH, SIGMA
import numpy as np
import tensorflow as tf
import cv2

class Encoder(object):
    def __init__(self):
        self.bin_centers = np.load("resources/bin_centers.npy")
        self.prior_weights = np.load("resources/weights.npy")
        self.knn = nn.NearestNeighbors(
            n_neighbors=5, algorithm='ball_tree').fit(self.bin_centers)
        self.pixel_idx = np.arange(IMAGE_HEIGHT * IMAGE_WIDTH)[:, np.newaxis]

    def soft_encode(self, img):
        assert(img.shape[0] == IMAGE_HEIGHT)
        assert(img.shape[1] == IMAGE_WIDTH)
        assert(img.shape[2] == 2)

        label = np.zeros((IMAGE_HEIGHT * IMAGE_WIDTH, 313))
        img = np.reshape(img, (-1, 2))
        distances, indices = self.knn.kneighbors(img, 5)
        weights = np.exp(-distances ** 2 / (2 * SIGMA ** 2))
        weights = weights / np.sum(weights, axis=1)[:, np.newaxis]
        label[self.pixel_idx, indices] = weights
        label = np.reshape(label, (IMAGE_HEIGHT, IMAGE_WIDTH, 313))

        return label

    def soft_encode_with_color_rebal(self, img):
        label = np.zeros((IMAGE_HEIGHT * IMAGE_WIDTH, 313))
        img = np.reshape(img, (-1, 2))
        distances, indices = self.knn.kneighbors(img, 5)

        weights = np.exp(-distances ** 2 / (2 * SIGMA ** 2))
        weights = weights / np.sum(weights, axis=1)[:, np.newaxis]

        label[self.pixel_idx, indices] = weights
        label = np.reshape(label.T, (313, IMAGE_HEIGHT, IMAGE_WIDTH))

        rebal_idx = indices[:, 0]
        rebal_weights = self.prior_weights[rebal_idx]
        rebal_weights = np.reshape(rebal_weights, (IMAGE_WIDTH, IMAGE_HEIGHT))
        rebal_label = label * rebal_weights
        rebal_label = rebal_label.T

        return rebal_label

    def decode(self, logits, l_imgs, ab_imgs):
        for i in range(ab_imgs.shape[0]):
            prbs = tf.nn.softmax(logits[i])
            # prbs = self.soft_encode(ab_imgs[i]) # uncomment to test soft encoding/decoding
            ab_img = np.reshape(np.dot(np.reshape(prbs, (-1, 313)), self.bin_centers), (IMAGE_HEIGHT, IMAGE_WIDTH, 2))
            l_img = np.reshape(l_imgs[i], (-1, IMAGE_HEIGHT, IMAGE_WIDTH))
            img = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
            img[:, :, 0] = l_img
            img[:, :, 1:] = ab_img
            img = ((img + [0, 128, 128]) / [255, 1, 1]) * [100, 1, 1]
            img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_LAB2BGR)
            cv2.imwrite("output/img{}.png".format(i), img)
            print("saving img{}.png".format(i))
