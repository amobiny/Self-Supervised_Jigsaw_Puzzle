import tensorflow as tf
import time

flags = tf.app.flags
flags.DEFINE_string('mode', 'train', 'train or test')

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

# hamming set
flags.DEFINE_boolean('generateHammingSet', False, 'Generate a new HammingSet')
flags.DEFINE_integer('hammingSetSize', 100, 'Hamming set size')
flags.DEFINE_string('selectionMethod', 'max', 'max or mean')
flags.DEFINE_string('hammingFileName', 'max_hamming_set_100.h5', 'Name of the file to be saved')


# jigsaw
flags.DEFINE_integer('numCrops', 9, 'The number of jigsaw-puzzle crops')
flags.DEFINE_integer('cellSize', 75, 'The dimensions of the jigsaw input')
flags.DEFINE_integer('tileSize', 64, 'The dimensions of the jigsaw input')
flags.DEFINE_integer('colorJitter', 2, 'Number of pixels for color jittering')
flags.DEFINE_integer('cropSize', 225, 'Size of the crop extracted from each input image')

# Directories
flags.DEFINE_string('run_name', 'run01', 'Run name')
flags.DEFINE_string('logdir', './Results/log_dir/', 'Logs directory')
flags.DEFINE_string('modeldir', './Results/model_dir/', 'Saved models directory')
flags.DEFINE_string('savedir', './Results/result/', 'Results saving directory')

flags.DEFINE_string('model_name', 'model', 'Model file name')
flags.DEFINE_integer('reload_step', 0, 'Reload step to continue training')


args = tf.app.flags.FLAGS
