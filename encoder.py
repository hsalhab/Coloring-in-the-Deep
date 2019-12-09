import sklearn.neighbors as nn
from hyperparameters import IMAGE_HEIGHT, IMAGE_WIDTH, SIGMA
import numpy as np
import tensorflow as tf
from skimage.color import lab2rgb, rgb2lab
from PIL import Image

class Encoder(object):
    def __init__(self):
        self.bin_centers = np.load("bin_centers.npy")
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

    def decode(self, logits, l_imgs, ab_imgs):
        for i in range(ab_imgs.shape[0]):
            prbs = tf.nn.softmax(logits[i])
            # prbs = self.soft_encode(ab_imgs[i]) # uncomment to test soft encoding/decoding
            ab_img = np.reshape(np.dot(np.reshape(prbs, (-1, 313)), self.bin_centers), (IMAGE_HEIGHT, IMAGE_WIDTH, 2))
            l_img = np.reshape(l_imgs[i], (-1, IMAGE_HEIGHT, IMAGE_WIDTH))
            img = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
            img[:, :, 0] = l_img
            img[:, :, 1:] = ab_img
            img = lab2rgb(img)
            img = (255 * np.clip(img, 0, 1)).astype('uint8')
            img_ = Image.fromarray(img)
            img_.save("output/img{}.png".format(i))
            print("saving img{}.png".format(i))
