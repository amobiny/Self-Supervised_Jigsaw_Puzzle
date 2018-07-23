from keras import initializers, layers
import keras.backend as K
import tensorflow as tf
import numpy as np
from keras.utils.conv_utils import conv_output_length
from layers.ops import update_routing


class ConvCapsuleLayer(layers.Layer):
    def __init__(self, kernel_size, num_caps, caps_dim, strides=1, padding='same', routings=3,
                 kernel_initializer='he_normal', **kwargs):
        super(ConvCapsuleLayer, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.num_caps = num_caps
        self.caps_dim = caps_dim
        self.strides = strides
        self.padding = padding
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) == 5, "The input Tensor should have shape=[None, input_height, input_width," \
                                      " input_num_capsule, input_num_atoms]"
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.num_in_caps = input_shape[3]
        self.in_caps_dim = input_shape[4]

        # Transform matrix
        self.W = self.add_weight(shape=[self.kernel_size, self.kernel_size,
                                        self.in_caps_dim, self.num_caps * self.caps_dim],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.b = self.add_weight(shape=[1, 1, self.num_caps, self.caps_dim],
                                 initializer=initializers.constant(0.1),
                                 name='b')

        self.built = True

    def call(self, input_tensor, training=None):
        input_transposed = tf.transpose(input_tensor, [3, 0, 1, 2, 4])
        input_shape = K.shape(input_transposed)
        input_tensor_reshaped = K.reshape(input_transposed, [
            input_shape[0] * input_shape[1], self.input_height, self.input_width, self.in_caps_dim])
        input_tensor_reshaped.set_shape((None, self.input_height, self.input_width, self.in_caps_dim))

        conv = K.conv2d(input_tensor_reshaped, self.W, (self.strides, self.strides),
                        padding=self.padding, data_format='channels_last')

        votes_shape = K.shape(conv)
        _, conv_height, conv_width, _ = conv.get_shape()

        votes = K.reshape(conv, [input_shape[1], input_shape[0], votes_shape[1], votes_shape[2],
                                 self.num_caps, self.caps_dim])
        votes.set_shape((None, self.num_in_caps, conv_height.value, conv_width.value,
                         self.num_caps, self.caps_dim))

        logit_shape = K.stack([input_shape[1], input_shape[0], votes_shape[1], votes_shape[2], self.num_caps])
        biases_replicated = K.tile(self.b, [conv_height.value, conv_width.value, 1, 1])

        activations = update_routing(votes=votes,
                                     biases=biases_replicated,
                                     logit_shape=logit_shape,
                                     num_dims=6,
                                     input_dim=self.num_in_caps,
                                     output_dim=self.num_caps,
                                     num_routing=self.routings)

        return activations

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-2]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_output_length(
                space[i],
                self.kernel_size,
                padding=self.padding,
                stride=self.strides,
                dilation=1)
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_caps, self.caps_dim)

    def get_config(self):
        config = {'kernel_size': self.kernel_size,
                  'num_capsule': self.num_caps,
                  'num_atoms': self.caps_dim,
                  'strides': self.strides,
                  'padding': self.padding,
                  'routings': self.routings,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer)}
        base_config = super(ConvCapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
