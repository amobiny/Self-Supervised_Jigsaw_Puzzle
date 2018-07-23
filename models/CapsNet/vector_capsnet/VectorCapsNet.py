import tensorflow as tf
from layers.Conv_Caps import ConvCapsuleLayer
from layers.FC_Caps import FCCapsuleLayer
from keras import layers


def VectorCapsNet(x, conf):
    # Layer 1: A 3D conv layer
    conv1 = layers.Conv2D(filters=64, kernel_size=5, strides=1,
                          padding='same', activation='relu', name='conv1')(x)

    # Reshape layer to be 1 capsule x caps_dim(=filters)
    _, H, W, C = conv1.get_shape()
    conv1_reshaped = layers.Reshape((H.value, W.value, 1, C.value))(conv1)

    # Layer 2: Convolutional Capsule
    primary_caps = ConvCapsuleLayer(kernel_size=5, num_caps=4, caps_dim=16, strides=2, padding='same',
                                    routings=3, name='primarycaps')(conv1_reshaped)

    # Layer 3: Convolutional Capsule
    # secondary_caps = ConvCapsuleLayer(kernel_size=5, num_caps=4, caps_dim=8, strides=2, padding='same',
    #                                   routings=3, name='secondarycaps')(primary_caps)
    _, H, W, D, dim = primary_caps.get_shape()
    sec_cap_reshaped = layers.Reshape((H.value * W.value * D.value, dim.value))(primary_caps)

    # Layer 4: Fully-connected Capsule
    digit_caps = FCCapsuleLayer(num_caps=conf.num_cls, caps_dim=conf.digit_caps_dim,
                                routings=3, name='secondarycaps')(sec_cap_reshaped)
    # [?, 10, 16]

    epsilon = 1e-9
    v_length = tf.sqrt(tf.reduce_sum(tf.square(digit_caps), axis=2, keep_dims=True) + epsilon)
    # [?, 10, 1]

    y_prob_argmax = tf.to_int32(tf.argmax(v_length, axis=1))
    # [?, 1]
    y_pred = tf.squeeze(y_prob_argmax)
    # [?] (predicted labels)
    y_pred_ohe = tf.one_hot(y_pred, depth=conf.num_cls)
    # [?, 10] (one-hot-encoded predicted labels)
