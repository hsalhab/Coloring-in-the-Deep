from math import floor
import tensorflow as tf
import numpy as np
from encoder import Encoder
from model import IC_Model
from preprocess import get_train_data
import hyperparameters as hp


def train():
    model = IC_Model()
    l_imgs, ab_imgs = get_train_data()
    encoder = Encoder()
    num_batches = floor(l_imgs.shape[0] / hp.BATCH_SIZE)
    for batch_i in range(num_batches):
        inputs_batch = l_imgs[batch_i * hp.BATCH_SIZE: (batch_i + 1) * hp.BATCH_SIZE]
        labels_batch = get_batch_labels(ab_imgs[batch_i * hp.BATCH_SIZE: (batch_i + 1) * hp.BATCH_SIZE], encoder)
        print("batch {} out of {}".format(batch_i, num_batches))
        with tf.GradientTape() as tape:
            logits = model.call(inputs_batch)
            loss = model.loss(logits, labels_batch)

        print("loss: {}".format(loss))
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def get_batch_labels(batch_ab, encoder):
    labels = []
    for img in batch_ab:
        label = encoder.soft_encode(img)
        labels.append(label)
    return np.asarray(labels)
