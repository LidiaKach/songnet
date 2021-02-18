from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Dropout, Input, concatenate, MaxPooling2D, AveragePooling2D, \
    GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import Lambda, Flatten, Dense


def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     conv_strides=(1, 1),
                     name=None):
    net = []

    if filters_1x1 > 0:
        conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same')(x)
        conv_1x1 = BatchNormalization(epsilon=0.00001)(conv_1x1)
        conv_1x1 = Activation('relu')(conv_1x1)
        net.append(conv_1x1)

    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same')(x)
    conv_3x3 = BatchNormalization(epsilon=0.00001)(conv_3x3)
    conv_3x3 = Activation('relu')(conv_3x3)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', strides=conv_strides)(conv_3x3)
    conv_3x3 = BatchNormalization(epsilon=0.00001)(conv_3x3)
    conv_3x3 = Activation('relu')(conv_3x3)
    net.append(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same')(x)
    conv_5x5 = BatchNormalization(epsilon=0.00001)(conv_5x5)
    conv_5x5 = Activation('relu')(conv_5x5)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', strides=conv_strides)(conv_5x5)
    conv_5x5 = BatchNormalization(epsilon=0.00001)(conv_5x5)
    conv_5x5 = Activation('relu')(conv_5x5)
    net.append(conv_5x5)

    pool_proj = MaxPooling2D((3, 3), strides=conv_strides, padding='same')(x)

    if filters_pool_proj > 0:
        pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same')(pool_proj)
        pool_proj = BatchNormalization(epsilon=0.00001)(pool_proj)
        pool_proj = Activation('relu')(pool_proj)

    net.append(pool_proj)

    output = concatenate(net, axis=3, name=name)

    return output


def faceRecoModel(input_shape, emb_size):
    """
    Implementation of the Inception model used for FaceNet

    Arguments:
    input_shape -- shape of the images of the dataset
    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape

    input_layer = Input(input_shape)

    x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), name='conv_1_7x7/2')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)

    x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), name='conv_2a_3x3/1')(x)
    x = BatchNormalization(epsilon=0.00001)(x)
    x = Activation('relu')(x)
    x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), name='conv_2b_3x3/1')(x)
    x = BatchNormalization(epsilon=0.00001)(x)
    x = Activation('relu')(x)

    x = MaxPooling2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)

    x = inception_module(x,
                         filters_1x1=64,
                         filters_3x3_reduce=96,
                         filters_3x3=128,
                         filters_5x5_reduce=16,
                         filters_5x5=32,
                         filters_pool_proj=32,
                         name='inception_3a')

    x = inception_module(x,
                         filters_1x1=64,
                         filters_3x3_reduce=96,
                         filters_3x3=128,
                         filters_5x5_reduce=32,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         name='inception_3b')

    x = inception_module(x,
                         filters_1x1=0,
                         filters_3x3_reduce=128,
                         filters_3x3=128,
                         filters_5x5_reduce=32,
                         filters_5x5=64,
                         filters_pool_proj=0,

                         name='inception_3c')

    x = inception_module(x,
                         filters_1x1=256,
                         filters_3x3_reduce=96,
                         filters_3x3=192,
                         filters_5x5_reduce=32,
                         filters_5x5=64,
                         filters_pool_proj=128,
                         name='inception_4a')

    x = inception_module(x,
                         filters_1x1=224,
                         filters_3x3_reduce=112,
                         filters_3x3=224,
                         filters_5x5_reduce=31,
                         filters_5x5=64,
                         filters_pool_proj=128,
                         name='inception_4b')

    x = inception_module(x,
                         filters_1x1=192,
                         filters_3x3_reduce=128,
                         filters_3x3=256,
                         filters_5x5_reduce=31,
                         filters_5x5=64,
                         filters_pool_proj=128,
                         name='inception_4c')

    x = inception_module(x,
                         filters_1x1=160,
                         filters_3x3_reduce=144,
                         filters_3x3=288,
                         filters_5x5_reduce=32,
                         filters_5x5=64,
                         filters_pool_proj=128,
                         name='inception_4d')

    x = inception_module(x,
                         filters_1x1=0,
                         filters_3x3_reduce=160,
                         filters_3x3=256,
                         filters_5x5_reduce=64,
                         filters_5x5=128,
                         filters_pool_proj=0,
                         conv_strides=(2, 2),
                         name='inception_4e')

    x = inception_module(x,
                         filters_1x1=384,
                         filters_3x3_reduce=192,
                         filters_3x3=384,
                         filters_5x5_reduce=48,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         name='inception_5a')

    x = inception_module(x,
                         filters_1x1=384,
                         filters_3x3_reduce=192,
                         filters_3x3=384,
                         filters_5x5_reduce=48,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         name='inception_5b')

    #x = GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)

    x = AveragePooling2D(pool_size=(33, 14), strides=(1, 1))(x)

    x = Flatten()(x)
    x = Dense(emb_size, name='dense_layer')(x)

    # L2 normalization
    x = Lambda(lambda x: K.l2_normalize(x, axis=1))(x)

    # Create model instance
    model = Model(inputs=input_layer, outputs=x, name='FaceRecoModel')
    return model
