import h5py
import random
import time
import numpy as np
from image_preprocessing import image_transform
import itertools
import warnings
import threading


class DataGenerator:
    """
    Class for a generator that reads in data from the HDF5 file, one batch at
    a time, converts it into the jigsaw, and then returns the data
    """

    def __init__(self, maxHammingSet, data_path='./prepare_data/COCO_2017_unlabeled.h5',
                 xDim=64, yDim=64, numChannels=3,
                 numCrops=9, batchSize=32, meanTensor=None, stdTensor=None):
        """
        meanTensor - rank 3 tensor of the mean for each pixel for all colour channels, used to normalize the data
        stdTensor - rank 3 tensor of the std for each pixel for all colour channels, used to normalize the data
        maxHammingSet - a
        """
        self.data_path = data_path
        self.xDim = xDim
        self.yDim = yDim
        self.numChannels = numChannels
        self.numCrops = numCrops
        self.batchSize = batchSize
        self.meanTensor = meanTensor.astype(np.float32)
        self.stdTensor = stdTensor.astype(np.float32)
        self.maxHammingSet = np.array(maxHammingSet, dtype=np.uint8)
        self.batch_counter()
        # Determine how many possible jigsaw puzzle arrangements there are
        self.numJigsawTypes = self.maxHammingSet.shape[0]
        # Use default options for JigsawCreator
        self.jigsawCreator = image_transform.JigsawCreator(maxHammingSet=maxHammingSet)

    def batch_counter(self):
        h5f = h5py.File(self.data_path, 'r')
        self.numTrainBatch = h5f['train_img'][:].shape[0]
        self.numValBatch = h5f['val_img'][:].shape[0]
        h5f.close()
        self.batchIndexTrain = 0
        self.batchIndexVal = 0

    def __data_generation_normalize(self, x):
        """
        Internal method used to help generate data, used when
        dataset - an HDF5 dataset (either train or validation)
        """
        x -= self.meanTensor
        x /= self.stdTensor
        # This implementation modifies each image individually
        y = np.empty(self.batchSize)
        # Python list of 4D numpy tensors for each channel
        X = [np.empty((self.batchSize, self.xDim, self.yDim, self.numChannels), np.float32)
             for _ in range(self.numCrops)]
        for image_num in range(self.batchSize):
            # Transform the image into its nine croppings
            single_image, y[image_num] = self.jigsawCreator.create_croppings(x[image_num])
            for image_location in range(self.numCrops):
                X[image_location][image_num, :, :, :] = single_image[:, :, :, image_location]
        return X, y

    def one_hot(self, y):
        """
        Returns labels in binary NumPy array
        """
        return np.array([[1 if y[i] == j else 0 for j in range(self.numJigsawTypes)]
                         for i in range(y.shape[0])])

    #  @threadsafe_generator
    def next_batch(self, mode='train'):
        """
        dataset - an HDF5 dataset (either train or validation)
        """
        if mode == 'train':
            self.batchIndexTrain += 1  # Increment the batch index
            h5f = h5py.File(self.data_path, 'r')
            x = h5f['train_img'][self.batchIndexTrain * self.batchSize:(self.batchIndexTrain + 1) * self.batchSize, ...]
            h5f.close()
            X, y = self.__data_generation_normalize(x)
            if self.batchIndexTrain == self.numTrainBatch:
                self.batchIndexTrain = 0
        elif mode == 'valid':
            self.batchIndexVal += 1  # Increment the batch index
            h5f = h5py.File(self.data_path, 'r')
            x = h5f['valid_img'][self.batchIndexVal * self.batchSize:(self.batchIndexVal + 1) * self.batchSize, ...]
            h5f.close()
            X, y = self.__data_generation_normalize(x)
            if self.batchIndexVal == self.numValBatch:
                self.batchIndexVal = 0
        return X, self.one_hot(y)

