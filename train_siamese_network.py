from songsearch.nn2 import faceRecoModel
from songsearch.base_model import base_model
from songsearch import config
from songsearch import utils
import os
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


emb_size = config.EMBEDDING_DIM

x_train, y_train = utils.load_data(os.path.join(os.getcwd(), config.DATA_PATH), "spect.npz")

alpha = 0.2
batch_size = config.BATCH_SIZE
epochs = config.EPOCHS
steps_per_epoch = 10

embedding_model = faceRecoModel(config.SPEC_SHAPE, config.EMBEDDING_DIM)
embedding_model.summary()
net = base_model(config.SPEC_SHAPE, embedding_model)

net.compile(loss=utils.identity_loss, optimizer='adam')

_ = net.fit(
    utils.data_generator(x_train, y_train, config.EMBEDDING_DIM, config.SPEC_SHAPE, batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=epochs, verbose=2,

)

net.save_weights(os.path.sep.join([config.MODEL_PATH, "model_weights_gpu.h5"]))
