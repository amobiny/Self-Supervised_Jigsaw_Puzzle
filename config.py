import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('mode', 'train', 'train or test')
flags.DEFINE_string('model', 'matrix_capsule', 'matrix_capsule or vector_capsule or alexnet')
flags.DEFINE_string('loss_type', 'spread', 'spread or margin or cross_entropy')

# Matrix Capsule architecture
flags.DEFINE_boolean('use_bias', True, 'Adds bias to init capsules')
flags.DEFINE_boolean('use_BN', True, 'Adds BN before conv1 layer')
flags.DEFINE_boolean('add_coords', True, 'Adds capsule coordinations')
flags.DEFINE_boolean('grad_clip', False, 'Adds gradient clipping to get rid of exploding gradient')
flags.DEFINE_boolean('L2_reg', True, 'Adds L2-regularization to all the network weights')
flags.DEFINE_float('lmbda', 5e-04, 'L2-regularization coefficient')
flags.DEFINE_boolean('add_decoder', False, 'Adds a fully connected decoder and reconstruction loss')
flags.DEFINE_integer('iter', 1, 'Number of EM-routing iterations')
flags.DEFINE_integer('A', 32, 'A in Figure 1 of the paper')
flags.DEFINE_integer('B', 32, 'B in Figure 1 of the paper')
flags.DEFINE_integer('C', 32, 'C in Figure 1 of the paper')
flags.DEFINE_integer('D', 32, 'D in Figure 1 of the paper')

# Training logs
flags.DEFINE_integer('max_epoch', 10000, 'maximum number of training epochs')
flags.DEFINE_integer('SUMMARY_FREQ', 100, 'Number of step to save summary')
flags.DEFINE_integer('VAL_FREQ', 1000, 'Number of step to evaluate the network on Validation data')

# Hyper-parameters
# For training
flags.DEFINE_integer('batchSize', 64, 'training batch size')
flags.DEFINE_integer('val_batch_size', 64, 'validation batch size')
flags.DEFINE_float('init_lr', 1e-3, 'Initial learning rate')
flags.DEFINE_float('lr_min', 1e-4, 'Minimum learning rate')

# data
flags.DEFINE_string('data_path', './prepare_data/COCO_2017_unlabeled.h5', 'Data path')
flags.DEFINE_boolean('data_augment', True, 'Adds augmentation to data')
flags.DEFINE_integer('max_angle', 40, 'Maximum rotation angle along each axis; when applying augmentation')
flags.DEFINE_integer('num_tr', 55000, 'Total number of training images')
flags.DEFINE_integer('height', 64, 'Input height size')
flags.DEFINE_integer('width', 64, 'Input width size')
flags.DEFINE_integer('depth', 32, 'Input depth size')
flags.DEFINE_integer('numChannels', 3, 'Input channel size')
flags.DEFINE_integer('numClasses', 10, 'Input channel size')


# Directories
flags.DEFINE_string('run_name', 'run01', 'Run name')
flags.DEFINE_string('logdir', './Results/log_dir/', 'Logs directory')
flags.DEFINE_string('modeldir', './Results/model_dir/', 'Saved models directory')
flags.DEFINE_string('savedir', './Results/result/', 'Results saving directory')

flags.DEFINE_string('model_name', 'model', 'Model file name')
flags.DEFINE_integer('reload_step', 0, 'Reload step to continue training')

args = tf.app.flags.FLAGS
