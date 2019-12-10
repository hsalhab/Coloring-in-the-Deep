from math import floor
import tensorflow as tf
import numpy as np
from encoder import Encoder
from model import IC_Model
from preprocess import get_batch, fetch_data
import hyperparameters as hp
import os


def train(model, encoder, manager, random_batches):
    num_batches = random_batches.shape[0]
    best_loss = float("inf")
    for i, batch_idx in enumerate(random_batches):
        print("batch {} out of {}".format(i, num_batches))
        l_imgs, ab_imgs = get_batch(batch_idx)
        labels = get_batch_labels(ab_imgs, encoder)
        with tf.GradientTape() as tape:
            logits = model.call(l_imgs)
            loss = model.loss_function(logits, labels)

        print("loss: {}".format(loss))
        
        if i > 100 and loss < best_loss:
            manager.save()
            best_loss = loss

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, encoder):
    l_imgs, ab_imgs = get_batch(0)
    logits = model.call(l_imgs)
    encoder.decode(logits, l_imgs, ab_imgs)


def get_batch_labels(batch_ab, encoder):
    labels = []
    for img in batch_ab:
        label = encoder.soft_encode(img)
        labels.append(label)
    return np.asarray(labels)


# Ensure the checkpoint directory exists
if not os.path.exists("./checkpoints"):
    os.makedirs("./checkpoints")

testing = True
model = IC_Model()
encoder = Encoder()
num_batches = floor(fetch_data() / hp.BATCH_SIZE)

# For saving/loading models
checkpoint_dir = './checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(model=model)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

if not os.path.exists("./output"):
    os.makedirs("./output")

if testing:
    # restores the latest checkpoint using from the manager
    checkpoint.restore(manager.latest_checkpoint).expect_partial()
    test(model, encoder)
else:
    # checkpoint.restore(manager.latest_checkpoint)
    epochs = 100
    for i in range(epochs):
        print("epoch {} out of {}".format(i, epochs))
        random_batches = np.arange(start=0, stop=num_batches, dtype=np.int32)
        np.random.shuffle(random_batches)
        train(model, encoder, manager, random_batches)
    test(model, encoder)

