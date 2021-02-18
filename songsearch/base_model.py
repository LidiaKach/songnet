import tensorflow as tf
from songsearch import utils


def base_model(input_shape, embedding_net):
    # Define the tensors for the three input images
    input_anchor = tf.keras.layers.Input(shape=input_shape)
    input_positive = tf.keras.layers.Input(shape=input_shape)
    input_negative = tf.keras.layers.Input(shape=input_shape)

    # Generate the encodings (feature vectors) for the three images
    embedding_anchor = embedding_net(input_anchor)
    embedding_positive = embedding_net(input_positive)
    embedding_negative = embedding_net(input_negative)

    # TripletLoss Layer
    loss = tf.keras.layers.Lambda(utils.triplet_loss)([embedding_anchor, embedding_positive, embedding_negative])

    # Connect the inputs with the outputs
    network_train = tf.keras.models.Model(inputs=[input_anchor, input_positive, input_negative], outputs=loss)

    # return the model
    return network_train
