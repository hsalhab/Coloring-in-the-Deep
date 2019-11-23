import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense


class IC_Model(tf.keras.Model):
    def __init__(self):
        super(IC_Model, self).__init__()


        self.learning_rate = 0.01
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.batch_size = 100


    @tf.function
    def call(self, inputs):
        """
        Do forward pass
        :param inputs: forward pass inputs
        :return:
        """

        pass


    def loss_function(self, prbs, labels):
        """
        Calculates the model loss after one forward pass

        :param prbs:  Probabilities tensor from forward pass
        :param labels:  Actual labels
        :return: the loss of the model as a tensor
        """
        # TODO: this needs to be changed after conversation with Ritchie
        pass

