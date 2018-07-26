import tensorflow as tf
from layers.Conv_Caps import ConvCapsuleLayer
from layers.FC_Caps import FCCapsuleLayer
from keras import layers


class VectorCapsNet:
    def __init__(self, conf, is_train=True):
        self.conf = conf
        self.summary_list = []
        self.is_train = is_train
        self.build()

    def build(self):
        self.summary_list = []
        self.conv = layers.Conv2D(filters=64, kernel_size=5, strides=2,
                                  padding='valid', activation='relu', name='conv1')
        self.conv_cap1 = ConvCapsuleLayer(kernel_size=3, num_caps=2, caps_dim=16, strides=2,
                                          padding='valid', routings=2, name='conv_cap1')
        self.conv_cap2 = ConvCapsuleLayer(kernel_size=3, num_caps=4, caps_dim=16, strides=2,
                                          padding='valid', routings=2, name='conv_cap2')
        self.conv_cap3 = ConvCapsuleLayer(kernel_size=3, num_caps=4, caps_dim=16, strides=2,
                                          padding='valid', routings=2, name='conv_cap3')
        self.fc_cap = FCCapsuleLayer(num_caps=self.conf.num_fc_caps, caps_dim=self.conf.fc_caps_dim,
                                     routings=2, name='fc_cap')

    def __call__(self, x, reuse=False):
        # Building network...
        with tf.variable_scope('CapsNet', reuse=reuse):
            # Layer 1: A 3D conv layer
            conv1 = self.conv(x)
            # Reshape layer to be 1 capsule x caps_dim(=filters)
            _, H, W, C = conv1.get_shape()
            conv1_reshaped = layers.Reshape((H.value, W.value, 1, C.value))(conv1)

            # Layer 2: Convolutional Capsule
            out_caps = self.conv_cap1(conv1_reshaped)

            # Layer 3: Convolutional Capsule
            out_caps = self.conv_cap2(out_caps)

            # Layer 4: Convolutional Capsule
            out_caps = self.conv_cap3(out_caps)

            _, H, W, D, dim = out_caps.get_shape()
            self.out_caps = layers.Reshape((H.value * W.value * D.value, dim.value))(out_caps)

            if not reuse:
                self.summary_list.append(tf.summary.histogram('conv/w', self.conv.weights[0]))
                self.summary_list.append(tf.summary.histogram('conv/b', self.conv.weights[1]))
                self.summary_list.append(tf.summary.histogram('convcap1/W', self.conv_cap1.W))
                self.summary_list.append(tf.summary.histogram('convcap2/W', self.conv_cap2.W))

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
