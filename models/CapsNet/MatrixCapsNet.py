import tensorflow as tf
from matrix_capsnet_ops.ops import conv_2d, capsules_init, capsule_conv, capsule_fc


class MatrixCapsNet:
    def __init__(self, conf, is_train=True):
        self.conf = conf
        self.summary_list = []
        self.is_train = is_train

    def __call__(self, x):
        # Building network...
        with tf.variable_scope('CapsNet'):
            net, summary = conv_2d(x, 5, 2, self.conf.A, 'CONV1', add_bias=self.conf.use_bias,
                                   add_reg=self.conf.L2_reg, batch_norm=self.conf.use_BN, is_train=self.is_train)
            # [?, 14, 14, A]
            self.summary_list.append(summary)

            pose, act, summary_list = capsules_init(net, 1, 1, OUT=self.conf.B, padding='VALID',
                                                    pose_shape=[4, 4], add_reg=self.conf.L2_reg,
                                                    use_bias=self.conf.use_bias, name='capsule_init')
            # [?, 14, 14, B, 4, 4], [?, 14, 14, B]
            for summary in summary_list:
                self.summary_list.append(summary)

            pose, act, summary_list = capsule_conv(pose, act, K=3, OUT=self.conf.C, stride=2, add_reg=self.conf.L2_reg,
                                                   iters=self.conf.iter, std=1, name='capsule_conv1')
            # [?, 6, 6, C, 4, 4], [?, 6, 6, C]
            for summary in summary_list:
                self.summary_list.append(summary)

            pose, act, summary_list = capsule_conv(pose, act, K=3, OUT=self.conf.D, stride=1, add_reg=self.conf.L2_reg,
                                                   iters=self.conf.iter, std=1, name='capsule_conv2')
            # [?, 4, 4, D, 4, 4], [?, 4, 4, D]
            for summary in summary_list:
                self.summary_list.append(summary)

            out_pose, out_act, summary_list = capsule_fc(pose, act, OUT=self.conf.num_cls, add_reg=self.conf.L2_reg,
                                                         iters=self.conf.iter, std=1, add_coord=self.conf.add_coords,
                                                         name='capsule_fc')
            # [?, num_cls, 4, 4], [?, num_cls]

            return out_act, out_pose, summary_list
