import os

# specify the shape of the inputs for our network
SPEC_SHAPE = (513, 221, 1)
# specify the batch size and number of epochs
BATCH_SIZE = 5
EPOCHS = 5

# define the path to the base output directory
BASE_OUTPUT = "output"
# use the base output path to derive the path to the serialized model
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
CLASSIFIER_PATH = os.path.sep.join([BASE_OUTPUT, "classifier"])

# define the path to the songs directory
SONG_PATH = "audio"

# define path to the data directory
DATA_PATH = "data"



EMBEDDING_DIM = 128