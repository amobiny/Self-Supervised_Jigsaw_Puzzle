import tensorflow as tf
from EM import matrix_capsules_em_routing
import numpy as np


def weight_variable(shape, name='W', std=0.01):
    """
    Create a weight variable with appropriate initialization
    :param name: weight name
    :param shape: weight shape
    :return: initialized weight variable
    """
    # initer = tf.contrib.layers.xavier_initializer(uniform=False)
    initer = tf.truncated_normal_initializer(stddev=std)
    return tf.get_variable(name, dtype=tf.float32,
                           shape=shape, initializer=initer)


def bias_variable(shape, name='b'):
    """
    Create a bias variable with appropriate initialization
    :param name: bias variable name
    :param shape: bias variable shape
    :return: initial bias variable
    """
    # initial = tf.constant(0., shape=shape, dtype=tf.float32)
    # return tf.get_variable(name, dtype=tf.float32,
    #                        initializer=initial)
    initial = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable(name, dtype=tf.float32, shape=shape,
                           initializer=initial)


def conv_2d(inputs, filter_size, stride, num_filters, name, padding='SAME', add_bias=False, add_reg=False,
            is_train=True, batch_norm=False, act_func=tf.nn.relu, std=0.01):
    """Create a convolution layer."""

    num_inChannel = inputs.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        if batch_norm:
            inputs = batch_norm_wrapper(inputs, is_train)
        shape = [filter_size, filter_size, num_inChannel, num_filters]
        weights = weight_variable(shape=shape, std=std)
        summary = []
        # summary = tf.summary.histogram('w', weights)
        layer = tf.nn.conv2d(input=inputs,
                             filter=weights,
                             strides=[1, stride, stride, 1],
                             padding=padding)

        if add_bias:
            biases = bias_variable([num_filters])
            layer += biases
        if add_reg:
            tf.add_to_collection('weights', weights)
        layer = act_func(layer)
    return layer, summary


def capsules_init(inputs, filter_size, stride, OUT, pose_shape, padding='VALID',
                  add_reg=False, use_bias=True, name=None):
    """This constructs a primary capsule layer from a regular convolution layer."""
    with tf.variable_scope(name):
        sum_list = []
        num_filters = OUT * pose_shape[0] * pose_shape[1]
        pose, w_summary = conv_2d(inputs, filter_size, stride, num_filters, 'pose_stacked', add_reg=add_reg,
                                  padding=padding, add_bias=use_bias, act_func=tf.identity)
        # sum_list.append(w_summary)
        poses_shape = pose.get_shape().as_list()
        pose = tf.reshape(pose, shape=[-1] + poses_shape[1:-1] + [OUT, pose_shape[0], pose_shape[1]], name='poses')
        activations, w_summary = conv_2d(inputs, filter_size, stride, OUT, 'activation',
                                         padding=padding, add_bias=use_bias, act_func=tf.sigmoid)
        sum_list.append(w_summary)
        a_summary = tf.summary.histogram('activations', activations)
        sum_list.append(a_summary)
        return pose, activations, sum_list


def capsule_conv(input_pose, input_act, K, OUT, stride, iters, std=0.01, add_reg=False, name=None):
    """This constructs a convolution capsule layer from a primary or convolution capsule layer."""

    _, H, W, IN, PH, PW = input_pose.get_shape().as_list()
    with tf.variable_scope(name):
        sum_list = []
        weights = weight_variable(name='pose_weight', shape=[K, K, IN, OUT, PH, PH], std=std)
        # w_summary = tf.summary.histogram('w', weights)
        hk_offsets = [[(h_offset + k_offset) for k_offset in range(0, K)] for h_offset in
                      range(0, W + 1 - K, stride)]
        wk_offsets = [[(w_offset + k_offset) for k_offset in range(0, K)] for w_offset in
                      range(0, H + 1 - K, stride)]
        # sum_list.append(w_summary)
        inputs_poses_patches = tf.transpose(tf.gather(tf.gather(input_pose, hk_offsets, axis=1), wk_offsets, axis=3),
                                            perm=[0, 1, 3, 2, 4, 5, 6, 7])
        # [N, OH, OW, KH, KW, IN, PH, PW]
        input_pose_patches = inputs_poses_patches[..., tf.newaxis, :, :]
        # to [N, OH, OW, KH, KW, IN, 1, PH, PW]
        input_pose_patches = tf.tile(input_pose_patches, [1, 1, 1, 1, 1, 1, OUT, 1, 1])
        # to [N, OH, OW, KH, KW, IN, OUT, PH, PW]
        votes = _matmul_broadcast(input_pose_patches, weights)
        # [N, OH, OW, KH, KW, IN, OUT, PH, PW]
        votes_shape = votes.get_shape().as_list()
        votes = tf.reshape(votes, [-1, votes_shape[1], votes_shape[2],
                                   votes_shape[3] * votes_shape[4] * votes_shape[5],
                                   votes_shape[6], votes_shape[7] * votes_shape[8]])
        # reshape into [N, OH, OW, KH x KW x IN, OUT, PH x PW]

        input_activation_patches = tf.transpose(tf.gather(tf.gather(input_act, hk_offsets, axis=1),
                                                          wk_offsets, axis=3), perm=[0, 1, 3, 2, 4, 5])
        # [N, OH, OW, KH, KW, I]
        act = tf.reshape(input_activation_patches, [-1, votes_shape[1], votes_shape[2],
                                                    votes_shape[3] * votes_shape[4] * votes_shape[5]])
        # [N, OH, OW, KH x KW x I]
        beta_v = weight_variable(shape=[1, 1, 1, votes_shape[6]], name='beta_v', std=0)
        beta_v_summary = tf.summary.histogram('beta_v', beta_v)
        sum_list.append(beta_v_summary)
        beta_a = weight_variable(shape=[1, 1, 1, votes_shape[6]], name='beta_a', std=0)
        beta_a_summary = tf.summary.histogram('beta_a', beta_a)
        # sum_list.append(beta_a_summary)
        if add_reg:
            tf.add_to_collection('weights', weights)
            tf.add_to_collection('weights', beta_v)
            tf.add_to_collection('weights', beta_a)
        out_pose, out_act = matrix_capsules_em_routing(votes, act, beta_v, beta_a, iters, name='EM')
        out_pose = tf.reshape(out_pose, [-1, votes_shape[1], votes_shape[2],
                                         votes_shape[6], votes_shape[7], votes_shape[8]])
        a_summary = tf.summary.histogram('activations', out_act)
        # sum_list.append(a_summary)
        return out_pose, out_act, sum_list


def capsule_fc(input_pose, input_act, OUT, iters, std=0.01, add_coord=True, add_reg=False, name=None):
    _, H, W, IN, PH, PW = input_pose.get_shape().as_list()
    sum_list = []
    with tf.variable_scope(name):
        weights = weight_variable(name='pose_weight', shape=[IN, OUT, PH, PH], std=std)
        # w_summary = tf.summary.histogram('w', weights)
        # sum_list.append(w_summary)
        input_pose_expansion = input_pose[..., tf.newaxis, :, :]
        # [N, H, W, I, 1, PH, PW]
        inputs_poses_expansion = tf.tile(input_pose_expansion, [1, 1, 1, 1, OUT, 1, 1])
        # [N, H, W, I, OUT, PH, PW]
        votes = _matmul_broadcast(inputs_poses_expansion, weights)
        # [N, H, W, I, OUT, PH, PW]
        votes_shape = votes.get_shape().as_list()
        votes = tf.reshape(votes, [-1] + votes_shape[1:-2] + [votes_shape[-2] * votes_shape[-1]])
        # [N, H, W, I, OUT, PH x PW]

        if add_coord:
            # coords = np.array([[[8., 8.], [12., 8.], [16., 8.], [24., 8.]],
            #           [[8., 12.], [12., 12.], [16., 12.], [24., 12.]],
            #           [[8., 16.], [12., 16.], [16., 16.], [24., 16.]],
            #           [[8., 24.], [12., 24.], [16., 24.], [24., 24.]]], dtype=np.float32) / 28.
            # coords = np.reshape(coords, newshape=[1, votes_shape[1], votes_shape[2], 1, 1, 2])
            # coords = np.tile(coords, [votes_shape[0], 1, 1, votes_shape[3], votes_shape[4], 1])
            # coord_add_op = tf.constant(coords, dtype=tf.float32)
            # votes = tf.concat([coord_add_op, votes], axis=-1)

            coordinate_offset_hh = tf.reshape((tf.range(H, dtype=tf.float32) + 0.50) / H, [1, H, 1, 1, 1])
            coordinate_offset_h0 = tf.constant(0.0, shape=[1, H, 1, 1, 1], dtype=tf.float32)
            coordinate_offset_h = tf.stack([coordinate_offset_hh, coordinate_offset_h0]
                                           + [coordinate_offset_h0 for _ in xrange(14)], axis=-1)

            coordinate_offset_ww = tf.reshape((tf.range(W, dtype=tf.float32) + 0.50) / W, [1, 1, W, 1, 1])
            coordinate_offset_w0 = tf.constant(0.0, shape=[1, 1, W, 1, 1], dtype=tf.float32)
            coordinate_offset_w = tf.stack([coordinate_offset_w0, coordinate_offset_ww]
                                           + [coordinate_offset_w0 for _ in xrange(14)], axis=-1)

            votes = votes + coordinate_offset_h + coordinate_offset_w
        votes = tf.reshape(votes, [votes_shape[0], votes_shape[1] * votes_shape[2] * votes_shape[3],
                                   votes_shape[4], -1])
        i_act = tf.reshape(input_act, [-1, H * W * IN])

        beta_v = weight_variable(shape=[1, OUT], name='beta_v', std=0)
        # beta_v_summary = tf.summary.histogram('beta_v', beta_v)
        # sum_list.append(beta_v_summary)
        beta_a = weight_variable(shape=[1, OUT], name='beta_a', std=0)
        # beta_a_summary = tf.summary.histogram('beta_a', beta_a)
        # sum_list.append(beta_a_summary)
        if add_reg:
            tf.add_to_collection('weights', weights)
            tf.add_to_collection('weights', beta_v)
            tf.add_to_collection('weights', beta_a)

        pose, out_act = matrix_capsules_em_routing(votes, i_act, beta_v, beta_a, iters, name='EM')
        # [N, O, PH x PW], [N, O]
        pose = tf.reshape(pose, [votes_shape[0], votes_shape[4], votes_shape[5], votes_shape[6]])
        # [N, O, PH, PW]
        # a_summary = tf.summary.histogram('activations', out_act)
        # sum_list.append(a_summary)

    return pose, tf.cast(out_act, tf.float32), sum_list


def _matmul_broadcast(x, y):
    """Compute x @ y, broadcasting over the first `N - 2` ranks.
  """
    return tf.reduce_sum(x[..., tf.newaxis] * y[..., tf.newaxis, :, :], axis=-2)


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
