from keras import initializers, layers
import keras.backend as K
import tensorflow as tf
import numpy as np
from keras.utils.conv_utils import conv_output_length

from layers.ops import squash


class FCCapsuleLayer(layers.Layer):
    def __init__(self, num_caps, caps_dim, routings=3,
                 kernel_initializer='he_normal', **kwargs):
        super(FCCapsuleLayer, self).__init__(**kwargs)
        self.num_caps = num_caps
        self.caps_dim = caps_dim
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) == 3, "The input Tensor should have shape=[None, num_in_caps, in_caps_dim]"
        self.num_in_caps = input_shape[1]
        self.in_caps_dim = input_shape[2]

        # Transform matrix
        self.W = self.add_weight(shape=[self.num_caps, self.num_in_caps,
                                        self.caps_dim, self.in_caps_dim],
                                 initializer=self.kernel_initializer,
                                 name='W')
        self.built = True

    def call(self, input_tensor, training=None):
        inputs_hat = self.get_predictions(input_tensor)
        b = tf.zeros(shape=[K.shape(inputs_hat)[0], self.num_caps, self.num_in_caps])

        assert self.routings > 0, 'routing should be > 0.'
        for i in range(self.routings):
            # c.shape=[batch_size, num_caps, num_in_caps]
            c = tf.nn.softmax(b, dim=1)
            activations = squash(K.batch_dot(c, inputs_hat, [2, 2]))  # [None, 10, 16]

        return activations

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_caps, self.caps_dim])

    def get_config(self):
        config = {
            'num_capsule': self.num_caps,
            'dim_capsule': self.caps_dim,
            'routings': self.routings
        }
        base_config = super(FCCapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_predictions(self, input_tensor):
        inputs_expand = K.expand_dims(input_tensor, 1)
        inputs_tiled = K.tile(inputs_expand, [1, self.num_caps, 1, 1])
        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled)
        return inputs_hat
