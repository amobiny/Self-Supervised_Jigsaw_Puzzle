import tensorflow as tf
from layers.Conv_Caps import ConvCapsuleLayer
from layers.FC_Caps import FCCapsuleLayer
from keras import layers
import keras.backend as K


class OrigCapsNet:
    def __init__(self, conf, is_train=True):
        self.conf = conf
        self.summary_list = []
        self.is_train = is_train
        self.build()

    def build(self):
        self.summary_list = []
        self.conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=2,
                                   padding='valid', activation='relu', name='conv1')
        self.conv2 = layers.Conv2D(filters=256, kernel_size=9, strides=2,
                                   padding='valid', activation='relu', name='primary_caps')
        self.fc_cap = FCCapsuleLayer(num_caps=self.conf.num_fc_caps, caps_dim=self.conf.fc_caps_dim,
                                     routings=2, name='fc_cap')

    def __call__(self, x, reuse=False):
        # Building network...
        with tf.variable_scope('CapsNet', reuse=reuse):
            conv1 = self.conv1(x)

            # Layer 2: Primary Capsule Layer; simply a 2D conv + reshaping
            primary_caps = self.conv2(conv1)
            _, H, W, dim = primary_caps.get_shape()
            num_caps = H.value * W.value * dim.value / self.conf.prim_caps_dim
            primary_caps_reshaped = layers.Reshape((num_caps, self.conf.prim_caps_dim))(primary_caps)
            self.out_caps = squash(primary_caps_reshaped)

            if not reuse:
                self.summary_list.append(tf.summary.histogram('conv1/w', self.conv1.weights[0]))
                self.summary_list.append(tf.summary.histogram('conv1/b', self.conv1.weights[1]))
                self.summary_list.append(tf.summary.histogram('conv2/w', self.conv2.weights[0]))
                self.summary_list.append(tf.summary.histogram('conv2/b', self.conv2.weights[1]))

            if self.conf.fc:
                # Layer 4: Fully-connected Capsule
                self.out_caps = self.fc_cap(self.out_caps)
                # [?, num_fc_caps, fc_caps_dim]
                self.summary_list.append(tf.summary.histogram('FC/W', self.fc_cap.W))

        return self.out_caps, self.summary_list

    # def mask(self):
    #     with tf.variable_scope('Masking'):
    #         self.y_pred = tf.to_int32(tf.argmax(self.v_length, axis=1))
    #         # [?] (predicted labels)
    #         y_pred_ohe = tf.one_hot(self.y_pred, depth=self.conf.hammingSetSize)
    #         # [?, 10] (one-hot-encoded predicted labels)
    #
    #         reconst_targets = tf.cond(self.is_train,  # condition
    #                                   lambda: self.y,  # if True (Training)
    #                                   lambda: y_pred_ohe,  # if False (Test)
    #                                   name="reconstruction_targets")
    #         # [?, 10]
    #         self.output_masked = tf.multiply(self.digit_caps, tf.expand_dims(reconst_targets, -1))
    #         # [?, 10, 16]
    #
    # def decoder(self):
    #     with tf.variable_scope('Decoder'):
    #         decoder_input = tf.reshape(self.output_masked, [-1, self.conf.num_cls * self.conf.digit_caps_dim])
    #         # [?, 160]
    #         fc1 = tf.layers.dense(decoder_input, self.conf.h1, activation=tf.nn.relu, name="FC1")
    #         # [?, 512]
    #         fc2 = tf.layers.dense(fc1, self.conf.h2, activation=tf.nn.relu, name="FC2")
    #         # [?, 1024]
    #         self.decoder_output = tf.layers.dense(fc2, self.conf.width * self.conf.height,
    #                                               activation=tf.nn.sigmoid, name="FC3")
    #         # [?, 784]


def squash(input_tensor, axis=-1):
    squared_norm = K.sum(K.square(input_tensor), axis, keepdims=True)
    scale = squared_norm / (1 + squared_norm) / K.sqrt(squared_norm + K.epsilon())
    return scale * input_tensor
