from keras.layers import (Dense, Input, Activation, Flatten, Conv2D,
                          MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, add)
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K























def residualMapping(inputTensor, filters):
    """
    Residual building block where input and output tensor are the same
    dimensions
    """
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(inputTensor)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = add([x, inputTensor])
    x = Activation('relu')(x)

    return x


def downsizeMapping(inputTensor, filters):
    """
    Residual building block where input tensor dimensions are halved, but
    feature map dimensions double
    """
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    x = Conv2D(filters, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(inputTensor)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    inputTensor = Conv2D(filters, (1, 1), strides=(2, 2), kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(inputTensor)
    x = add([x, inputTensor])
    x = Activation('relu')(x)

    return x


def ResNet34(inputShape):
    """
    Creates a stack of layers equivalent to ResNet-34 architecture up until the
    average pool and 1000-d fully connected layer.
    Assumes that the input is a square patch, so that the output is 1x1x512
    regardless of input dimensions (using average pooling)
    """
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    #  if (inputShape[0] != inputShape[1]):
    #      warnings.warn("Image input shape was non-square", Warning)

    inputTensor = Input(inputShape)

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(inputTensor)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = downsizeMapping(x, 64)
    x = residualMapping(x, 64)
    x = residualMapping(x, 64)

    x = downsizeMapping(x, 128)
    x = residualMapping(x, 128)
    x = residualMapping(x, 128)
    x = residualMapping(x, 128)

    x = downsizeMapping(x, 256)
    x = residualMapping(x, 256)
    x = residualMapping(x, 256)
    x = residualMapping(x, 256)
    x = residualMapping(x, 256)
    x = residualMapping(x, 256)

    x = downsizeMapping(x, 512)
    x = residualMapping(x, 512)
    x = residualMapping(x, 512)

    x = GlobalAveragePooling2D()(x)

    model = Model(inputTensor, x, name='ResNet34')
    return model