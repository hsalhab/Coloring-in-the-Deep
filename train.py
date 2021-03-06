from math import floor
import tensorflow as tf
import numpy as np
from encoder import Encoder
from model import IC_Model
from preprocess import get_batch, fetch_data, shuffle_data
import hyperparameters as hp
import os


def train(model, encoder, manager, num_batches):
    best_loss = float("inf")
    for batch_idx in range(num_batches):
        print("batch {} out of {}".format(batch_idx, num_batches))
        l_imgs, ab_imgs = get_batch(batch_idx)
        labels = get_batch_labels(ab_imgs, encoder)
        with tf.GradientTape() as tape:
            logits = model.call(l_imgs)
            loss = model.loss_function(logits, labels)

        print("loss: {}".format(loss))
        
        if epoch > 100 and loss < best_loss:
            manager.save()
            best_loss = loss

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, encoder, epoch, num_batches):
    if not os.path.exists("./output/{}".format(epoch)):
        os.makedirs("./output/{}".format(epoch))

    if not os.path.exists("./output/{}/0".format(epoch)):
        os.makedirs("./output/{}/0".format(epoch))
    l_imgs, ab_imgs = get_batch(0)
    logits = model.call(l_imgs)
    encoder.decode(logits, l_imgs, ab_imgs, epoch, 0)

    batch_id = np.random.randint(1, num_batches-1)
    if not os.path.exists("./output/{}/{}".format(epoch, batch_id)):
        os.makedirs("./output/{}/{}".format(epoch, batch_id))
    l_imgs, ab_imgs = get_batch(batch_id)
    logits = model.call(l_imgs)
    encoder.decode(logits, l_imgs, ab_imgs, epoch, batch_id)


def get_batch_labels(batch_ab, encoder):
    labels = []
    for img in batch_ab:
        label = encoder.soft_encode_with_color_rebal(img)
        labels.append(label)
    return np.asarray(labels)


# Ensure the checkpoint directory exists
if not os.path.exists("./checkpoints"):
    os.makedirs("./checkpoints")

testing = False
model = IC_Model()
encoder = Encoder()
num_batches = int(floor(fetch_data() / hp.BATCH_SIZE))
num_batches = 3

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
    epochs = hp.EPOCHS
    for epoch in range(epochs):
        print("epoch {} out of {}".format(epoch, epochs))
        shuffle_data()
        train(model, encoder, manager, num_batches)
        test(model, encoder, epoch, num_batches)
    manager.save()


