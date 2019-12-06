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
    ab_img = tf.constant(ab_images)
    width = 0
    height = 0
    label = tf.zeros((width * height, 313))

    (dists, indices) = self.nbrs.kneighbors(
        abimg.view(abimg.shape[0], -1).t(), self.NN)
    dists = torch.from_numpy(dists).float().cuda()
    indexes = torch.from_numpy(indexes).cuda()

    weights = torch.exp(-dists ** 2 / (2 * self.sigma ** 2)).cuda()
    weights = weights / torch.sum(weights, dim=1).view(-1, 1)

    pixel_indexes = torch.Tensor.long(torch.arange(
        start=0, end=abimg.shape[1] * abimg.shape[2])[:, np.newaxis])
    pixel_indexes = pixel_indexes.cuda()
    label[pixel_indexes, indexes] = weights
    label = label.t().contiguous().view(313, w, h)

    rebal_indexes = indexes[:, 0]
    rebal_weights = self.weights[rebal_indexes]
    rebal_weights = rebal_weights.view(w, h)
    rebal_label = rebal_weights * label

    return rebal_label