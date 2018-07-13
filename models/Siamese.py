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
import tensorflow as tf
from models.loss_ops import cross_entropy_loss
from ops import *
from AlexNet import AlexNet
import numpy as np
from DataLoader.DataGenerator import DataGenerator
import os


class Siamese(object):

    def __init__(self, sess, conf, hamming_set):
        self.sess = sess
        self.conf = conf
        self.HammingSet = hamming_set
        self.input_shape = [None, conf.tileSize, conf.tileSize, conf.numChannels, conf.numCrops]
        self.is_train = tf.Variable(True, trainable=False, dtype=tf.bool)
        self.x, self.y, self.keep_prob = self.create_placeholders()
        self.inference()
        self.configure_network()

    def create_placeholders(self):
        with tf.name_scope('Input'):
            x = tf.placeholder(tf.float32, self.input_shape, name='x_input')
            y = tf.placeholder(tf.float32, shape=(None, self.conf.hammingSetSize), name='y_input')
            keep_prob = tf.placeholder(tf.float32)
        return x, y, keep_prob

    def inference(self):
        # Build the Network
        with tf.variable_scope('Siamese') as scope:
            Siamese_out = []
            x = tf.unstack(self.x, axis=-1)
            for i in range(self.conf.numCrops):
                Siamese_out.append(AlexNet(x[i], self.keep_prob, self.is_train))
                if i < self.conf.numCrops:
                    scope.reuse_variables()
        net = tf.concat(Siamese_out, axis=1)
        net = fc_layer(net, 4096, 'FC2', is_train=self.is_train, batch_norm=True, use_relu=True)
        net = dropout(net, self.keep_prob)
        self.logits = fc_layer(net, self.conf.hammingSetSize, 'FC3',
                               is_train=self.is_train, batch_norm=True, use_relu=False)

    def accuracy_func(self):
        with tf.name_scope('Accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.mean_accuracy, self.mean_accuracy_op = tf.metrics.mean(accuracy)

    def loss_func(self):
        with tf.name_scope('Loss'):
            self.total_loss = cross_entropy_loss(self.y, self.logits)
            self.mean_loss, self.mean_loss_op = tf.metrics.mean(self.total_loss)

    def configure_network(self):
        self.loss_func()
        self.accuracy_func()
        with tf.name_scope('Optimizer'):
            with tf.name_scope('Learning_rate_decay'):
                global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                              trainable=False)
                steps_per_epoch = self.conf.num_tr // self.conf.batchSize
                learning_rate = tf.train.exponential_decay(self.conf.init_lr,
                                                           global_step,
                                                           steps_per_epoch,
                                                           0.97,
                                                           staircase=True)
                self.learning_rate = tf.maximum(learning_rate, self.conf.lr_min)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.total_loss, global_step=global_step)
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
        # recon_img = tf.reshape(self.decoder_output, shape=(-1, self.conf.height, self.conf.width, self.conf.channel))
        summary_list = [tf.summary.scalar('Loss/total_loss', self.mean_loss),
                        # tf.summary.image('original', self.x),
                        # tf.summary.image('reconstructed', recon_img),
                        tf.summary.scalar('Accuracy/average_accuracy', self.mean_accuracy)]
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
                feed_dict = {self.x: x_batch, self.y: y_batch, self.keep_prob: 0.5}
                if train_step % self.conf.SUMMARY_FREQ == 0:
                    _, _, _, summary = self.sess.run([self.train_op,
                                                      self.mean_loss_op,
                                                      self.mean_accuracy_op,
                                                      self.merged_summary], feed_dict=feed_dict)
                    loss, acc = self.sess.run([self.mean_loss, self.mean_accuracy])
                    global_step = (epoch-1) * self.data_reader.numTrainBatch + train_step
                    self.save_summary(summary, global_step, mode='train')
                    print('step: {0:<6}, train_loss= {1:.4f}, train_acc={2:.01%}'.format(train_step, loss, acc))
                else:
                    self.sess.run([self.train_op, self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)
            self.evaluate(epoch)

    def evaluate(self, epoch):
        self.is_train = False
        self.sess.run(tf.local_variables_initializer())
        for step in range(self.data_reader.numValBatch):
            x_val, y_val = self.data_reader.generate(mode='valid')
            feed_dict = {self.x: x_val, self.y: y_val, self.keep_prob: 1}
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
            feed_dict = {self.x: x_test, self.y: y_test, self.keep_prob: 1}
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
        checkpoint_path = os.path.join(self.conf.modeldir+self.conf.run_name, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=epoch)

    def reload(self, epoch):
        checkpoint_path = os.path.join(self.conf.modeldir+self.conf.run_name, self.conf.model_name)
        model_path = checkpoint_path + '-' + str(epoch)
        if not os.path.exists(model_path + '.meta'):
            print('----> No such checkpoint found', model_path)
            return
        print('----> Restoring the model...')
        self.saver.restore(self.sess, model_path)
        print('----> Model-{} successfully restored'.format(epoch))
