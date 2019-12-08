from math import floor
import tensorflow as tf
import numpy as np
from encoder import Encoder
from model import IC_Model
from preprocess import get_train_data
import hyperparameters as hp
import os


def train(model, l_imgs, ab_imgs, manager):
    encoder = Encoder()
    num_batches = floor(l_imgs.shape[0] / hp.BATCH_SIZE)
    best_loss = float("inf")
    print(l_imgs.shape)
    for batch_i in range(num_batches):
        print("batch {} out of {}".format(batch_i, num_batches))
        inputs_batch = l_imgs[batch_i * hp.BATCH_SIZE: (batch_i + 1) * hp.BATCH_SIZE]
        labels_batch = get_batch_labels(ab_imgs[batch_i * hp.BATCH_SIZE: (batch_i + 1) * hp.BATCH_SIZE], encoder)
        with tf.GradientTape() as tape:
            logits = model.call(inputs_batch)
            loss = model.loss_function(logits, labels_batch)

        print("loss: {}".format(loss))
        
        if batch_i > 100 and loss < best_loss:
            manager.save()
            best_loss = loss

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def get_batch_labels(batch_ab, encoder):
    labels = []
    for img in batch_ab:
        label = encoder.soft_encode(img)
        labels.append(label)
    return np.asarray(labels)


# Ensure the checkpoint directory exists
if not os.path.exists("./checkpoints"):
    os.makedirs("./checkpoints")

test = False
model = IC_Model()
l_imgs, ab_imgs = get_train_data()

# For saving/loading models
checkpoint_dir = './checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(model=model)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

if test:
    # restores the latest checkpoint using from the manager
    if not os.path.exists("./output"):
        os.makedirs("./output")
    checkpoint.restore(manager.latest_checkpoint)
    np.random.shuffle(l_imgs)
    test(model, l_imgs[:10])
else:
    train(model, l_imgs, ab_imgs, manager)
