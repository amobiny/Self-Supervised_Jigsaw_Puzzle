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
                 xDim=64, yDim=64, numChannels=3, numCrops=9, cropSize=225, cellSize=75,
                 tileSize=64, colorJitter=2, batchSize=32, meanTensor=None, stdTensor=None):
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
        self.cropSize = cropSize
        self.cellSize = cellSize
        self.tileSize = tileSize
        self.colorJitter = colorJitter
        self.batchSize = batchSize
        self.meanTensor = meanTensor.astype(np.float32)
        self.stdTensor = stdTensor.astype(np.float32)
        self.maxHammingSet = np.array(maxHammingSet, dtype=np.uint8)
        self.batch_counter()
        self.numJigsawTypes = self.maxHammingSet.shape[0]
        self.numPermutations = self.maxHammingSet.shape[0]

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
            # Transform the image into its nine crops
            single_image, y[image_num] = self.create_croppings(x[image_num])
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

    def create_croppings(self, image):
        """
        Take in a 3D numpy array (256x256x3) and a 4D numpy array containing 9 "jigsaw" puzzles.
        Dimensions of the output array is 64 (height) x 64 (width) x 3 (colour channels) x 9(each cropping)

        The 3x3 grid is numbered as follows:
        0    1    2
        3    4    5
        6    7    8
        """
        # Jitter the colour channel
        image = self.color_channel_jitter(image)

        y_dim, x_dim = image.shape[:2]
        # Have the x & y coordinate of the crop
        crop_x = random.randrange(x_dim - self.cropSize)
        crop_y = random.randrange(y_dim - self.cropSize)
        # Select which image ordering we'll use from the maximum hamming set
        perm_index = random.randrange(self.numPermutations)
        final_crops = np.zeros((self.tileSize, self.tileSize, 3, 9), dtype=np.float32)
        for row in range(3):
            for col in range(3):
                x_start = crop_x + col * self.cellSize + random.randrange(self.cellSize - self.tileSize)
                y_start = crop_y + row * self.cellSize + random.randrange(self.cellSize - self.tileSize)
                # Put the crop in the list of pieces randomly according to the number picked
                final_crops[:, :, :, self.maxHammingSet[perm_index, row * 3 + col]] = \
                    image[y_start:y_start + self.tileSize, x_start:x_start + self.tileSize, :]
        return final_crops, perm_index

    def color_channel_jitter(self, image):
        """
        Takes in a 3D numpy array and then jitters the colour channels by
        between -2 and 2 pixels (to deal with overfitting to chromatic
        aberations).
        Input - a WxHx3 numpy array
        Output - a (W-4)x(H-4)x3 numpy array (3 colour channels for RGB)
        """
        # Determine the dimensions of the array, minus the crop around the border
        # of 4 pixels (threshold margin due to 2 pixel jitter)
        x_dim = image.shape[0] - self.colorJitter * 2
        y_dim = image.shape[1] - self.colorJitter * 2
        # Determine the jitters in all directions
        R_xjit = random.randrange(self.colorJitter * 2 + 1)
        R_yjit = random.randrange(self.colorJitter * 2 + 1)
        # Seperate the colour channels
        return_array = np.empty((x_dim, y_dim, 3), np.float32)
        for colour_channel in range(3):
            return_array[:, :, colour_channel] = \
                image[R_xjit:x_dim +R_xjit, R_yjit:y_dim + R_yjit, colour_channel]
        return return_array
