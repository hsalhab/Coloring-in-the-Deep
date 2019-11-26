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

        self.conv1_1 = Conv2D(64, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same')
        self.conv1_2 = Conv2D(64, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same', strides=(2, 2))
        self.batch1 = BatchNormalization()

        self.conv2_1 = Conv2D(128, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same')
        self.conv2_2 = Conv2D(128, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same', strides=(2, 2))
        self.batch2 = BatchNormalization()

        self.conv3_1 = Conv2D(256, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same')
        self.conv3_2 = Conv2D(256, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same')
        self.conv3_3 = Conv2D(256, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same', strides=(2, 2))
        self.batch3 = BatchNormalization()

        self.conv4_1 = Conv2D(512, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same')
        self.conv4_2 = Conv2D(512, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same')
        self.conv4_3 = Conv2D(512, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same')
        self.batch4 = BatchNormalization()

        self.conv5_1 = Conv2D(512, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same', dilation_rate=2)
        self.conv5_2 = Conv2D(512, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same', dilation_rate=2)
        self.conv5_3 = Conv2D(512, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same', dilation_rate=2)
        self.batch5 = BatchNormalization()

        self.conv6_1 = Conv2D(512, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same', dilation_rate=2)
        self.conv6_2 = Conv2D(512, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same', dilation_rate=2)
        self.conv6_3 = Conv2D(512, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same', dilation_rate=2)
        self.batch6 = BatchNormalization()

        self.conv7_1 = Conv2D(256, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same')
        self.conv7_2 = Conv2D(256, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same')
        self.conv7_3 = Conv2D(256, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same')
        self.batch7 = BatchNormalization()

        self.upsample = UpSampling2D(size=(2, 2))
        self.conv8_1 = Conv2D(128, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same')
        self.conv8_2 = Conv2D(128, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same')
        self.conv8_3 = Conv2D(128, kernel_size=hp.KERNEL_SIZE, activation='relu', padding='same')
        self.batch8 = BatchNormalization()

        self.final = Conv2D(hp.NUM_CLASSES, kernel_size=1, activation='softmax', padding='same')


    @tf.function
    def call(self, inputs):
        """
        Do forward pass
        :param inputs: forward pass inputs
        :return:
        """

        hidden = self.conv1_1(inputs)
        hidden = self.conv1_2(hidden)
        hidden = self.batch1(hidden)

        hidden = self.conv2_1(hidden)
        hidden = self.conv2_2(hidden)
        hidden = self.batch2(hidden)

        hidden = self.conv3_1(hidden)
        hidden = self.conv3_2(hidden)
        hidden = self.conv3_3(hidden)
        hidden = self.batch3(hidden)

        hidden = self.conv4_1(hidden)
        hidden = self.conv4_2(hidden)
        hidden = self.conv4_3(hidden)
        hidden = self.batch4(hidden)

        hidden = self.conv5_1(hidden)
        hidden = self.conv5_2(hidden)
        hidden = self.conv5_3(hidden)
        hidden = self.batch5(hidden)

        hidden = self.conv6_1(hidden)
        hidden = self.conv6_2(hidden)
        hidden = self.conv6_3(hidden)
        hidden = self.batch6(hidden)

        hidden = self.conv7_1(hidden)
        hidden = self.conv7_2(hidden)
        hidden = self.conv7_3(hidden)
        hidden = self.batch7(hidden)

        hidden = self.upsample(hidden)
        hidden = self.conv8_1(hidden)
        hidden = self.conv8_2(hidden)
        hidden = self.conv8_3(hidden)
        hidden = self.batch8(hidden)

        prbs = self.final(hidden)

        return prbs


    def loss_function(self, prbs, labels):
        """
        Calculates the model loss after one forward pass

        :param prbs:  Probabilities tensor from forward pass
        :param labels:  Actual labels
        :return: the loss of the model as a tensor
        """
        # TODO: soft-encode labels before passing into loss_function
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, prbs))


