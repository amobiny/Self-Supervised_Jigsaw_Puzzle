import tensorflow as tf
from layers.Conv_Caps import ConvCapsuleLayer
from layers.FC_Caps import FCCapsuleLayer
from keras import layers
from layers.ops import squash


class CapsNet:

    def __init__(self, x, is_train):
        self.x = x
        self.is_train = is_train

    def build_network(self):
        with tf.variable_scope('CapsNet'):
            # Layer 1: A 3D conv layer
            conv1 = layers.Conv3D(filters=256, kernel_size=9, strides=1,
                                  padding='valid', activation='relu', name='conv1')(self.x)

            # Layer 2: Primary Capsule Layer; simply a 3D conv + reshaping
            primary_caps = layers.Conv3D(filters=256, kernel_size=9, strides=2,
                                         padding='valid', activation='relu', name='primary_caps')(conv1)
            _, H, W, D, dim = primary_caps.get_shape()
            primary_caps_reshaped = layers.Reshape((H.value * W.value * D.value, dim.value))(primary_caps)
            caps1_output = squash(primary_caps_reshaped)
            # [?, 512, 256]
            # Layer 3: Digit Capsule Layer; Here is where the routing takes place
            digitcaps_layer = FCCapsuleLayer(num_caps=self.conf.num_cls, caps_dim=self.conf.digit_caps_dim,
                                             routings=3, name='digit_caps')
            self.digit_caps = digitcaps_layer(caps1_output)  # [?, 2, 16]
            u_hat = digitcaps_layer.get_predictions(caps1_output)  # [?, 2, 512, 16]
            u_hat_shape = u_hat.get_shape().as_list()
            self.img_s = int(round(u_hat_shape[2] ** (1. / 3)))
            self.u_hat = layers.Reshape(
                (self.conf.num_cls, self.img_s, self.img_s, self.img_s, 1, self.conf.digit_caps_dim))(u_hat)
            # self.u_hat = tf.transpose(u_hat, perm=[1, 0, 2, 3, 4, 5, 6])
            # u_hat: [?, 2, 8, 8, 8, 1, 16]
            self.decoder()
