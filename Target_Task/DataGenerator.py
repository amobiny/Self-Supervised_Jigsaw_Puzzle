import random
import scipy
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import h5py


class DataLoader(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.augment = cfg.data_augment
        self.max_angle = cfg.max_angle
        self.batch_size = cfg.batch_size
        self.num_tr = cfg.num_tr
        if self.cfg.dim == 2:
            self.data_path = '/home/cougarnet.uh.edu/amobiny/Desktop/Lung_Nodule_data/Lung_Nodule_2d.h5'
        elif self.cfg.dim == 3:
            self.data_path = '/home/cougarnet.uh.edu/amobiny/Desktop/Lung_Nodule_data/Lung_Nodule.h5'
        h5f = h5py.File(self.data_path, 'r')
        x_train = h5f['X_train'][:]
        y_train = h5f['Y_train'][:]
        h5f.close()
        self.mean = np.mean(x_train, axis=0)
        self.std = np.std(x_train, axis=0)
        self.x_train, self.y_train = self.preprocess(x_train, y_train, one_hot=cfg.one_hot)

    def next_batch(self, start=None, end=None, mode='train'):
        if mode == 'train':
            x = self.x_train[start:end]
            y = self.y_train[start:end]
            if self.augment:
                x = random_rotation_2d(x, self.cfg.max_angle)
        elif mode == 'valid':
            x = self.x_valid[start:end]
            y = self.y_valid[start:end]
        return x, y

    def get_validation(self):
        h5f = h5py.File(self.data_path, 'r')
        x_valid = h5f['X_valid'][:]
        y_valid = h5f['Y_valid'][:]
        h5f.close()
        self.x_valid, self.y_valid = self.preprocess(x_valid, y_valid, one_hot=self.cfg.one_hot)

    def randomize(self):
        """ Randomizes the order of data samples and their corresponding labels"""
        permutation = np.random.permutation(self.y_train.shape[0])
        self.x_train = self.x_train[permutation, :, :, :]
        self.y_train = self.y_train[permutation]

    def preprocess(self, x, y, normalize='standard', one_hot=True):
        x = np.maximum(np.minimum(x, 4096.), 0.)
        if normalize == 'standard':
            x = (x - self.mean) / self.std
        elif normalize == 'unity_based':
            x /= 4096.
        x = x.reshape((-1, self.cfg.height, self.cfg.width, self.cfg.depth, self.cfg.channel)).astype(np.float32)
        if one_hot:
            y = (np.arange(self.cfg.num_cls) == y[:, None]).astype(np.float32)
        return x, y


def random_rotation_2d(batch, max_angle):
    """ Randomly rotate an image by a random angle (-max_angle, max_angle).
    Arguments:
    max_angle: `float`. The maximum rotation angle.
    Returns:
    batch of rotated 2D images
    """
    size = batch.shape
    batch = np.squeeze(batch)
    batch_rot = np.zeros(batch.shape)
    for i in range(batch.shape[0]):
        if bool(random.getrandbits(1)):
            image = np.squeeze(batch[i])
            angle = random.uniform(-max_angle, max_angle)
            batch_rot[i] = scipy.ndimage.interpolation.rotate(image, angle, mode='nearest', reshape=False)
        else:
            batch_rot[i] = batch[i]
    return batch_rot.reshape(size)