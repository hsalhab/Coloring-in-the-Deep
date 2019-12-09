import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, UpSampling2D
import hyperparameters as hp


class IC_Model(tf.keras.Model):
    def __init__(self):
        super(IC_Model, self).__init__()

        # TODO: SGD might work better than Adam
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.LEARNING_RATE)
        self.batch_size = hp.BATCH_SIZE

        self.model = tf.keras.Sequential([
            Conv2D(64, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same'),
            Conv2D(64, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same', strides=(2, 2)),
            BatchNormalization(),

            Conv2D(128, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same'),
            Conv2D(128, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same', strides=(2, 2)),
            BatchNormalization(),

            Conv2D(256, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same'),
            Conv2D(256, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same'),
            Conv2D(256, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same', strides=(2, 2)),
            BatchNormalization(),

            Conv2D(512, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same'),
            Conv2D(512, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same'),
            Conv2D(512, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same'),
            BatchNormalization(),

            Conv2D(512, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same', dilation_rate=2),
            Conv2D(512, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same', dilation_rate=2),
            Conv2D(512, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same', dilation_rate=2),
            BatchNormalization(),

            Conv2D(512, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same', dilation_rate=2),
            Conv2D(512, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same', dilation_rate=2),
            Conv2D(512, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same', dilation_rate=2),
            BatchNormalization(),

            Conv2D(256, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same'),
            Conv2D(256, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same'),
            Conv2D(256, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same'),
            BatchNormalization(),

            UpSampling2D(size=(2, 2)),
            Conv2D(128, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same'),
            Conv2D(128, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same'),
            Conv2D(128, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same'),
            BatchNormalization(),

            UpSampling2D(size=(4, 4)),
            Conv2D(hp.NUM_CLASSES, kernel_size=1, padding='same')
        ])


    @tf.function
    def call(self, inputs):
        """
        Do forward pass
        :param inputs: forward pass inputs
        :return:
        """

        logits = self.model(inputs)
        return logits


    def loss_function(self, logits, labels):
        """
        Calculates the model loss after one forward pass

        :param prbs:  Probabilities tensor from forward pass
        :param labels:  Actual labels
        :return: the loss of the model as a tensor
        """
        # TODO: soft-encode labels before passing into loss_function
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))


