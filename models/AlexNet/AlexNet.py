"""
Copyright 2017-2022 Department of Electrical and Computer Engineering
University of Houston, TX/USA
**********************************************************************************
Author:   Aryan Mobiny
Date:     8/1/2017
Comments: Includes function which creates the 3D ResNet with 50 layer.
        For more information regarding the structure of the network,
        please refer to Table 1 of the original paper:
        "Deep Residual Learning for Image Recognition"
**********************************************************************************
"""

from ops import conv_2d, flatten_layer, fc_layer, dropout, max_pool, lrn


def AlexNet(X, keep_prob, is_train):
    net = conv_2d(X, 7, 2, 96, 'CONV1', trainable=True)
    net = lrn(net)
    net = max_pool(net, 3, 2, 'MaxPool1')
    net = conv_2d(net, 5, 2, 256, 'CONV2', trainable=True)
    net = lrn(net)
    net = max_pool(net, 3, 2, 'MaxPool2')
    net = conv_2d(net, 3, 1, 384, 'CONV3', trainable=True)
    net = conv_2d(net, 3, 1, 384, 'CONV4', trainable=True)
    net = conv_2d(net, 3, 1, 256, 'CONV5', trainable=True)
    net = max_pool(net, 3, 2, 'MaxPool3')
    layer_flat = flatten_layer(net)
    net = fc_layer(layer_flat, 512, 'FC1', trainable=True, use_relu=True)
    net = dropout(net, keep_prob)
    return net


def AlexNet_target_task(X, keep_prob, num_cls):
    net = conv_2d(X, 7, 2, 96, 'CONV1', trainable=False)
    net = lrn(net)
    net = max_pool(net, 3, 2, 'MaxPool1')
    net = conv_2d(net, 5, 2, 256, 'CONV2', trainable=False)
    net = lrn(net)
    net = max_pool(net, 3, 2, 'MaxPool2')
    net = conv_2d(net, 3, 1, 384, 'CONV3', trainable=False)
    net = conv_2d(net, 3, 1, 384, 'CONV4', trainable=False)
    net = conv_2d(net, 3, 1, 256, 'CONV5', trainable=False)
    net = max_pool(net, 3, 2, 'MaxPool3')
    layer_flat = flatten_layer(net)
    net = fc_layer(layer_flat, 512, 'FC_1', trainable=True, use_relu=True)
    net = dropout(net, keep_prob)
    net = fc_layer(net, num_cls, 'FC_2', trainable=True, use_relu=False)
    return net