"""
Copyright 2017-2022 Department of Electrical and Computer Engineering
University of Houston, TX/USA
**********************************************************************************
Author:   Aryan Mobiny
Date:     8/1/2017
Comments: ResNet with 50 convolutional layer for classifying the 3D lung nodule images.
This network is almost similar to the one with 50 layer used in the original
paper: "Deep Residual Learning for Image Recognition"
**********************************************************************************
"""
import numpy as np
from DataLoader.DataGenerator import DataGenerator
import os
import tensorflow as tf
from models.CapsNet.loss_ops import margin_loss, spread_loss
from models.CapsNet.matrix_capsnet.ops import capsule_fc
from models.CapsNet.vector_capsnet.layers.FC_Caps import FCCapsuleLayer


class SiameseCapsNet(object):
    def __init__(self, sess, conf, hamming_set):
        self.sess = sess
        self.conf = conf
        self.summary_list = []
        self.HammingSet = hamming_set
        self.input_shape = [conf.batchSize, conf.tileSize, conf.tileSize, conf.numChannels, conf.numCrops]
        self.is_train = tf.Variable(True, trainable=False, dtype=tf.bool)
        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                           trainable=False)
        self.x, self.y = self.create_placeholders()
        self.inference()
        self.configure_network()

    def create_placeholders(self):
        with tf.name_scope('Input'):
            x = tf.placeholder(tf.float32, self.input_shape, name='x_input')
            y = tf.placeholder(tf.float32, shape=(None, self.conf.hammingSetSize), name='y_input')
        return x, y

    def inference(self):
        # Build the Network
        if self.conf.model == 'matrix_capsule':
            from models.CapsNet.matrix_capsnet.MatrixCapsNet import MatrixCapsNet
            Network = MatrixCapsNet(self.conf, self.is_train)
            with tf.variable_scope('Siamese', reuse=tf.AUTO_REUSE):
                act_list = []
                pose_list = []
                x = tf.unstack(self.x, axis=-1)
                for i in range(self.conf.numCrops):
                    act, pose, summary_list = Network(x[i])
                    act_list.append(act)
                    pose_list.append(pose)
                # self.summary_list.append(summary_list)
                if self.conf.fc:
                    dim = np.sqrt(self.conf.numCrops).astype(int)
                    act = tf.reshape(tf.concat(act_list, axis=1), [self.conf.batchSize, dim, dim, -1])
                    pose = tf.reshape(tf.concat(pose_list, axis=1), [self.conf.batchSize, dim, dim, -1, 4, 4])
                else:
                    act = tf.concat(act_list, axis=-1)
                    pose = tf.concat(pose_list, axis=3)
                out_pose, self.out_act, summary_list = capsule_fc(pose, act, OUT=self.conf.hammingSetSize,
                                                                  add_reg=self.conf.L2_reg,
                                                                  iters=self.conf.iter, std=1,
                                                                  add_coord=self.conf.add_coords,
                                                                  name='capsule_fc2')
            self.y_pred = tf.to_int32(tf.argmax(self.out_act, axis=1))

        elif self.conf.model == 'vector_capsule':
            from models.CapsNet.vector_capsnet.VectorCapsNet import VectorCapsNet
            Network = VectorCapsNet(self.conf, self.is_train)
            reuse = False
            with tf.variable_scope('Siamese', reuse=reuse):
                out_caps_list = []
                x = tf.unstack(self.x, axis=-1)
                for i in range(self.conf.numCrops):
                    out_caps = Network(x[i], reuse=reuse)
                    out_caps_list.append(out_caps)
                    reuse = True
                out_caps = tf.concat(out_caps_list, axis=1)
                self.out_caps = FCCapsuleLayer(num_caps=self.conf.hammingSetSize, caps_dim=self.conf.out_caps_dim,
                                               routings=3, name='fc_caps')(out_caps)
                # [?, hammingSetSize, out_caps_dim]
                epsilon = 1e-9
                self.v_length = tf.squeeze(tf.sqrt(tf.reduce_sum(tf.square(self.out_caps),
                                                                 axis=2, keep_dims=True) + epsilon), axis=-1)
                # [?, hammingSetSize]
                self.y_pred = tf.to_int32(tf.argmax(self.v_length, axis=1))

    def accuracy_func(self):
        with tf.name_scope('Accuracy'):
            correct_prediction = tf.equal(tf.to_int32(tf.argmax(self.y, axis=1)), self.y_pred)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.mean_accuracy, self.mean_accuracy_op = tf.metrics.mean(accuracy)

    def loss_func(self):
        with tf.variable_scope('Loss'):
            if self.conf.loss_type == 'margin':
                loss = margin_loss(self.y, self.v_length, self.conf)
                self.summary_list.append(tf.summary.scalar('margin', loss))
            elif self.conf.loss_type == 'spread':
                self.generate_margin()
                loss = spread_loss(self.y, self.out_act, self.margin, 'spread_loss')
                self.summary_list.append(tf.summary.scalar('spread_loss', loss))
            if self.conf.L2_reg:
                with tf.name_scope('l2_loss'):
                    l2_loss = tf.reduce_sum(self.conf.lmbda * tf.stack([tf.nn.l2_loss(v)
                                                                        for v in tf.get_collection('weights')]))
                    loss += l2_loss
                self.summary_list.append(tf.summary.scalar('l2_loss', l2_loss))
            if self.conf.add_decoder:
                with tf.variable_scope('Reconstruction_Loss'):
                    orgin = tf.reshape(self.x, shape=(-1, self.conf.height * self.conf.width * self.conf.channel))
                    squared = tf.square(self.decoder_output - orgin)
                    self.recon_err = tf.reduce_mean(squared)
                    self.total_loss = loss + self.conf.alpha * self.conf.width * self.conf.height * self.recon_err
                    self.summary_list.append(tf.summary.scalar('reconstruction_loss', self.recon_err))
                    recon_img = tf.reshape(self.decoder_output,
                                           shape=(-1, self.conf.height, self.conf.width, self.conf.channel))
            else:
                self.total_loss = loss
            self.mean_loss, self.mean_loss_op = tf.metrics.mean(self.total_loss)

            if self.conf.add_decoder:
                self.summary_list.append(tf.summary.image('reconstructed', recon_img))
                self.summary_list.append(tf.summary.image('original', self.x))

    def generate_margin(self):
        # margin schedule
        # margin increase from 0.2 to 0.9 after margin_schedule_epoch_achieve_max
        margin_schedule_epoch_achieve_max = 10.0
        self.margin = tf.train.piecewise_constant(tf.cast(self.global_step, dtype=tf.int32),
                                                  boundaries=[int(self.NUM_STEPS_PER_EPOCH *
                                                                  margin_schedule_epoch_achieve_max * x / 7)
                                                              for x in xrange(1, 8)],
                                                  values=[x / 10.0 for x in range(2, 10)])

    def configure_network(self):
        self.NUM_STEPS_PER_EPOCH = int(self.conf.N / self.conf.batchSize)
        self.loss_func()
        self.accuracy_func()

        with tf.name_scope('Optimizer'):
            with tf.name_scope('Learning_rate_decay'):
                learning_rate = tf.train.exponential_decay(self.conf.init_lr,
                                                           self.global_step,
                                                           decay_steps=self.NUM_STEPS_PER_EPOCH,
                                                           decay_rate=0.9,
                                                           staircase=True)
                self.learning_rate = tf.maximum(learning_rate, self.conf.lr_min)
            self.summary_list.append(tf.summary.scalar('learning_rate', self.learning_rate))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            """Compute gradient."""
            grads = optimizer.compute_gradients(self.total_loss)
            # grad_check = [tf.check_numerics(g, message='Gradient NaN Found!') for g, _ in grads if g is not None] \
            #              + [tf.check_numerics(self.total_loss, message='Loss NaN Found')]
            """Apply graident."""
            # with tf.control_dependencies(grad_check):
            #     update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            #     with tf.control_dependencies(update_ops):
            """Add graident summary"""
            for grad, var in grads:
                self.summary_list.append(tf.summary.histogram(var.name, grad))
            if self.conf.grad_clip:
                """Clip graident."""
                grads = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in grads]
            """NaN to zero graident."""
            grads = [(tf.where(tf.is_nan(grad), tf.zeros(grad.shape), grad), var) for grad, var in grads]
            self.train_op = optimizer.apply_gradients(grads, global_step=self.global_step)
        self.sess.run(tf.global_variables_initializer())
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=1000)
        self.train_writer = tf.summary.FileWriter(self.conf.logdir + self.conf.run_name + '/train/', self.sess.graph)
        self.valid_writer = tf.summary.FileWriter(self.conf.logdir + self.conf.run_name + '/valid/')
        self.configure_summary()
        print('*' * 50)
        print('Total number of trainable parameters: {}'.
              format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
        print('*' * 50)

    def configure_summary(self):
        summary_list = [tf.summary.scalar('Loss/total_loss', self.mean_loss),
                        tf.summary.scalar('Accuracy/average_accuracy', self.mean_accuracy)] + self.summary_list
        self.merged_summary = tf.summary.merge(summary_list)

    def save_summary(self, summary, step, mode):
        # print('----> Summarizing at step {}'.format(step))
        if mode == 'train':
            self.train_writer.add_summary(summary, step)
        elif mode == 'valid':
            self.valid_writer.add_summary(summary, step)
        self.sess.run(tf.local_variables_initializer())

    def train(self):
        self.sess.run(tf.local_variables_initializer())
        self.best_validation_accuracy = 0
        self.data_reader = DataGenerator(self.conf, self.HammingSet)
        if self.conf.reload_step > 0:
            self.reload(self.conf.reload_step)
            print('*' * 50)
            print('----> Continue Training from step #{}'.format(self.conf.reload_step))
            print('*' * 50)
        else:
            print('*' * 50)
            print('----> Start Training')
            print('*' * 50)
        for epoch in range(1, self.conf.max_epoch):
            # self.data_reader.randomize()
            self.is_train = True
            for train_step in range(self.data_reader.numTrainBatch):
                x_batch, y_batch = self.data_reader.generate(mode='train')
                feed_dict = {self.x: x_batch, self.y: y_batch}
                if train_step % self.conf.SUMMARY_FREQ == 0:
                    _, _, _, summary = self.sess.run([self.train_op,
                                                      self.mean_loss_op,
                                                      self.mean_accuracy_op,
                                                      self.merged_summary], feed_dict=feed_dict)
                    loss, acc = self.sess.run([self.mean_loss, self.mean_accuracy])
                    global_step = (epoch - 1) * self.data_reader.numTrainBatch + train_step
                    self.save_summary(summary, global_step, mode='train')
                    print('step: {0:<6}, train_loss= {1:.4f}, train_acc={2:.2f}%'.format(train_step, loss, acc*100))
                else:
                    self.sess.run([self.train_op, self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)
            self.evaluate(epoch)

    def evaluate(self, epoch):
        self.is_train = False
        self.sess.run(tf.local_variables_initializer())
        for step in range(self.data_reader.numValBatch):
            x_val, y_val = self.data_reader.generate(mode='valid')
            feed_dict = {self.x: x_val, self.y: y_val}
            self.sess.run([self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)

        summary_valid = self.sess.run(self.merged_summary, feed_dict=feed_dict)
        valid_loss, valid_acc = self.sess.run([self.mean_loss, self.mean_accuracy])
        self.save_summary(summary_valid, epoch * self.data_reader.numTrainBatch, mode='valid')
        if valid_acc > self.best_validation_accuracy:
            self.best_validation_accuracy = valid_acc
            self.save(epoch)
            improved_str = '(improved)'
        else:
            improved_str = ''
        print('-' * 20 + 'Validation' + '-' * 20)
        print('After {0} epoch: val_loss= {1:.4f}, val_acc={2:.01%} {3}'.
              format(epoch, valid_loss, valid_acc, improved_str))
        print('-' * 50)

    def test(self, epoch_num):
        self.reload(epoch_num)
        self.data_reader = DataGenerator(self.conf, self.HammingSet)
        self.is_train = False
        self.sess.run(tf.local_variables_initializer())
        for step in range(self.data_reader.numTestBatch):
            x_test, y_test = self.data_reader.generate(mode='test')
            feed_dict = {self.x: x_test, self.y: y_test}
            self.sess.run([self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)
        test_loss, test_acc = self.sess.run([self.mean_loss, self.mean_accuracy])
        print('-' * 18 + 'Test Completed' + '-' * 18)
        print('test_loss= {0:.4f}, test_acc={1:.01%}'.
              format(test_loss, test_acc))
        print('-' * 50)

    def save(self, epoch):
        print('*' * 50)
        print('----> Saving the model after epoch #{0}'.format(epoch))
        print('*' * 50)
        checkpoint_path = os.path.join(self.conf.modeldir + self.conf.run_name, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=epoch)

    def reload(self, epoch):
        checkpoint_path = os.path.join(self.conf.modeldir + self.conf.run_name, self.conf.model_name)
        model_path = checkpoint_path + '-' + str(epoch)
        if not os.path.exists(model_path + '.meta'):
            print('----> No such checkpoint found', model_path)
            return
        print('----> Restoring the model...')
        self.saver.restore(self.sess, model_path)
        print('----> Model-{} successfully restored'.format(epoch))
