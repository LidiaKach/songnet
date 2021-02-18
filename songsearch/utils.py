from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import random
import os
import pickle
from songsearch import config


def create_batch(batch_size, x_input, y_input, input_shape):
    h, w, c = input_shape
    anchors = np.zeros((batch_size, h, w, c))
    positives = np.zeros((batch_size, h, w, c))
    negatives = np.zeros((batch_size, h, w, c))

    for i in range(batch_size):
        index = random.randint(0, x_input.shape[0] - 1)
        anc = x_input[index]
        y = y_input[index]
        indices_for_pos = np.squeeze(np.where(y_input == y))
        indices_for_neg = np.squeeze(np.where(y_input != y))

        pos = x_input[indices_for_pos[random.randint(0, len(indices_for_pos) - 1)]]
        neg = x_input[indices_for_neg[random.randint(0, len(indices_for_neg) - 1)]]

        anchors[i] = anc
        positives[i] = pos
        negatives[i] = neg

    return [anchors, positives, negatives]


def data_generator(x_input, y_input, emb_size, input_shape, batch_size=256):
    while True:
        x = create_batch(batch_size, x_input, y_input, input_shape)
        y = np.zeros((batch_size, 3 * emb_size))
        yield x, y


def identity_loss(y_true, y_pred):
    return K.mean(y_pred)


def triplet_loss(x, alpha=0.2):
    # Triplet Loss function.
    anchor, positive, negative = x
    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor - positive), axis=1)
    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor - negative), axis=1)
    # compute loss
    basic_loss = pos_dist - neg_dist + alpha
    loss = K.maximum(basic_loss, 0.0)
    return loss


def load_data(folder, filename):
    data = np.load(os.path.sep.join([folder, filename]))
    x, y = data.files
    x = data[x]
    y = data[y]

    x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)

    x = x.astype('float32')
    x = tf.keras.utils.normalize(x, axis=-1, order=2)
    map_dict = {}

    for token, value in enumerate(np.unique(y)):
        map_dict[value] = token
    with open('map_dict_y.pickle', 'wb') as f:
        pickle.dump(map_dict, f)

    for value, token in map_dict.items():
        y[y == value] = token

    y = y.astype(np.int32)

    return x, y

def reconstruct_net(recomodel, weights, input_shape, emb_size):
    input_sng = tf.keras.layers.Input(shape=input_shape)
    model = recomodel(input_shape, emb_size)
    output = model(input_sng)
    trained_model = tf.keras.models.Model(inputs=input_sng, outputs=output)
    trained_model.load_weights(os.path.sep.join([config.MODEL_PATH, weights]))
    return trained_model
