import tensorflow as tf
from DataLoader.DataGenerator import DataGenerator
from config import args
import os
from models.AlexNet.AlexNet import AlexNet_target_task
from models.AlexNet.loss_ops import cross_entropy_loss

epoch = 1
checkpoint_path = os.path.join(args.modeldir + args.run_name, args.model_name)
model_path = checkpoint_path + '-' + str(epoch)
if not os.path.exists(args.modeldir + args.run_name):
    os.makedirs(args.modeldir + args.run_name)
if not os.path.exists(args.logdir + args.run_name):
    os.makedirs(args.logdir + args.run_name)
if not os.path.exists(args.savedir + args.run_name):
    os.makedirs(args.savedir + args.run_name)


def save_summary(summa, step, mode):
    # print('----> Summarizing at step {}'.format(step))
    if mode == 'train':
        train_writer.add_summary(summa, step)
    elif mode == 'valid':
        valid_writer.add_summary(summa, step)
    sess.run(tf.local_variables_initializer())


input_shape = [None, 225, 225, 3]
x = tf.placeholder(tf.float32, input_shape, name='x_input')
y = tf.placeholder(tf.float32, [None, args.numClasses], name='y_input')
keep_prob = tf.placeholder(tf.float32)

with tf.variable_scope('Siamese') as scope:
    logits = AlexNet_target_task(x, keep_prob, args.numClasses)

with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    mean_accuracy, mean_accuracy_op = tf.metrics.mean(accuracy)

with tf.name_scope('Loss'):
    total_loss = cross_entropy_loss(y, logits)
    mean_loss, mean_loss_op = tf.metrics.mean(total_loss)

with tf.name_scope('Optimizer'):
    with tf.name_scope('Learning_rate_decay'):
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)
        steps_per_epoch = args.conf.num_tr // args.batchSize
        learning_rate = tf.train.exponential_decay(args.init_lr,
                                                   global_step,
                                                   steps_per_epoch,
                                                   0.97,
                                                   staircase=True)
        learning_rate = tf.maximum(learning_rate, args.lr_min)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(total_loss, global_step=global_step)

not_restore = ['Siamese/FC1/W:0', 'Siamese/FC1/b:0']
restore_var = [v for v in tf.all_variables() if v.name not in not_restore]
# Keep only the variables, whose name is not in the not_restore list.
saver1 = tf.train.Saver(var_list=restore_var, max_to_keep=1000)
saver2 = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=1000)

summary_list = [tf.summary.scalar('Loss/total_loss', mean_loss),
                tf.summary.scalar('Accuracy/average_accuracy', mean_accuracy)]
merged_summary = tf.summary.merge(summary_list)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(args.logdir + args.run_name + '/train/', sess.graph)
    valid_writer = tf.summary.FileWriter(args.logdir + args.run_name + '/valid/')
    saver1.restore(sess, model_path)
    data_reader = DataGenerator(args)
    num_train_batch = int(data_reader.y_train.shape[0] / args.batch_size)
    num_val_batch = int(data_reader.y_valid.shape[0] / args.val_batch_size)
    for epoch in range(args.max_epoch):
        data_reader.randomize()
        for train_step in range(num_train_batch):
            start = train_step * args.batch_size
            end = (train_step + 1) * args.batch_size
            global_step = epoch * num_train_batch + train_step
            x_batch, y_batch = data_reader.next_batch(start, end)
            feed_dict = {x: x_batch, y: y_batch, keep_prob: 0.5}
            if train_step % args.SUMMARY_FREQ == 0:
                _, _, _, summary = sess.run([train_op,
                                             mean_loss_op,
                                             mean_accuracy_op,
                                             merged_summary], feed_dict=feed_dict)
                loss, acc = sess.run([mean_loss, mean_accuracy])
                save_summary(summary, global_step, mode='train')
                print('step: {0:<6}, train_loss= {1:.4f}, train_acc={2:.01%}'.format(train_step, loss, acc))
            else:
                sess.run([train_op, mean_loss_op, mean_accuracy_op], feed_dict=feed_dict)
