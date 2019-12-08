import sklearn.neighbors as nn
from hyperparameters import IMAGE_HEIGHT, IMAGE_WIDTH, SIGMA
import numpy as np


class Encoder(object):
    def __init__(self):
        bin_centers = np.load("bin_centers.npy")
        self.knn = nn.NearestNeighbors(
            n_neighbors=5, algorithm='ball_tree').fit(bin_centers)
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
