"""
Copyright 2017-2022 Department of Electrical and Computer Engineering
University of Houston, TX/USA
**********************************************************************************
Author:   Aryan Mobiny
Date:     8/1/2017
Comments: Includes functions for defining the 3D-ResNet layers
**********************************************************************************
"""

import tensorflow as tf


def weight_variable(name, shape):
    """Create a weight variable with appropriate initialization."""
    initer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('W_' + name, dtype=tf.float32,
                           shape=shape, initializer=initer)


def bias_variable(name, shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b_' + name, dtype=tf.float32,
                           initializer=initial)


def conv_2d(inputs, filter_size, stride, num_filters, name,
            is_train=True, batch_norm=False, add_reg=False, use_relu=True):
    """Create a convolution layer."""

    num_inChannel = inputs.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        shape = [filter_size, filter_size, num_inChannel, num_filters]
        weights = weight_variable(name, shape=shape)
        tf.summary.histogram('W', weights)
        layer = tf.nn.conv2d(input=inputs,
                             filter=weights,
                             strides=[1, stride, stride, 1],
                             padding="SAME")
        if batch_norm:
            layer = batch_norm_wrapper(layer, is_train)
        else:
            biases = bias_variable(name, [num_filters])
            layer += biases
        if use_relu:
            layer = tf.nn.relu(layer)
        if add_reg:
            tf.add_to_collection('weights', weights)
    return layer


def flatten_layer(layer):
    """Flattens the output of the convolutional layer to prepare it to be fed in to the fully-connected layer"""
    with tf.variable_scope('Flatten_layer'):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat


def fc_layer(bottom, out_dim, name, is_train=True, batch_norm=False, add_reg=False, use_relu=True):
    """Create a fully connected layer"""
    in_dim = bottom.get_shape()[1]
    with tf.variable_scope(name):
        weights = weight_variable(name, shape=[in_dim, out_dim])
        tf.summary.histogram('W', weights)
        layer = tf.matmul(bottom, weights)
        if batch_norm:
            layer = batch_norm_wrapper(layer, is_train)
        else:
            biases = bias_variable(name, [out_dim])
            layer += biases
        if use_relu:
            layer = tf.nn.relu(layer)
        if add_reg:
            tf.add_to_collection('weights', weights)
    return layer


def max_pool(x, ksize, stride, name):
    """Create a max pooling layer."""
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding="SAME",
                          name=name)


def avg_pool(x, ksize, stride, name):
    """Create an average pooling layer."""
    return tf.nn.avg_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding="VALID",
                          name=name)


def dropout(x, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob)


def batch_norm_wrapper(inputs, is_training, decay=0.999, epsilon=1e-3):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        if len(inputs.get_shape().as_list()) == 4:  # For convolutional layers
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
        else:  # For fully-connected layers
            batch_mean, batch_var = tf.nn.moments(inputs, [0])
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)


def bottleneck_block(x, block_name, is_train,
                     s1, k1, nf1, name1,
                     s2, k2, nf2, name2,
                     s3, k3, nf3, name3,
                     s4, k4, name4, first_block=False):
    with tf.variable_scope(block_name):
        # Convolutional Layer 1
        layer_conv1 = conv_2d(inputs=x, filter_size=k1, stride=s1, num_filters=nf1, name=name1,
                              is_train=is_train, batch_norm=True, add_reg=False, use_relu=True)

        # Convolutional Layer 2
        layer_conv2 = conv_2d(inputs=layer_conv1, filter_size=k2, stride=s2, num_filters=nf2, name=name2,
                              is_train=is_train, batch_norm=True, add_reg=False, use_relu=True)

        # Convolutional Layer 3
        layer_conv3 = conv_2d(inputs=layer_conv2, filter_size=k3, stride=s3, num_filters=nf3, name=name3,
                              is_train=is_train, batch_norm=True, add_reg=False, use_relu=False)
        if first_block:
            shortcut = conv_2d(inputs=x, filter_size=k4, stride=s4, num_filters=nf3, name=name4,
                               is_train=is_train, batch_norm=True, add_reg=False, use_relu=False)
            assert (
                shortcut.get_shape().as_list() == layer_conv3.get_shape().as_list()), "Tensor sizes of the two branches are not matched!"
            res = shortcut + layer_conv3
        else:
            res = layer_conv3 + x
            assert (
                x.get_shape().as_list() == layer_conv3.get_shape().as_list()), "Tensor sizes of the two branches are not matched!"
    return tf.nn.relu(res)
