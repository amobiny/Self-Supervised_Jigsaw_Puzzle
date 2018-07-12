import h5py
from keras.models import Model
from keras.layers import Dense, Dropout, Concatenate, Input
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import optimizers
from DataLoader.DataGenerator import DataGenerator
from hamming_set.generate_hamming_set import hamming_set
from models.Residual_Network import ResNet34
from config import args
import numpy as np
from time import strftime, localtime
import os


def contextFreeNetwork(args):
    """
    Implemented non-siamese
    tileSize - The dimensions of the jigsaw input
    numPuzzles - the number of jigsaw puzzles

    returns a keras model
    """
    inputShape = (args.tileSize, args.tileSize, args.numChannels)
    modelInputs = [Input(inputShape) for _ in range(args.numCrops)]
    sharedLayer = ResNet34(inputShape)
    sharedLayers = [sharedLayer(inputTensor) for inputTensor in modelInputs]
    x = Concatenate()(sharedLayers)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(args.hammingSetSize, activation='softmax')(x)
    mod = Model(inputs=modelInputs, outputs=x)
    return mod


if args.generateHammingSet:
    hamming_set(args.numCrops, args.hammingSetSize,
                args.selectionMethod, args.hammingFilePath)

h5f = h5py.File('./hamming_set/' + args.hammingFileName, 'r')
HammingSet = np.array(h5f['max_hamming_set'])
h5f.close()

model = contextFreeNetwork(args)

dataGenerator = DataGenerator(args, HammingSet)

# Output all data from a training session into a dated folder
outputPath = './model_data/{}'.format(strftime('%b_%d_%H:%M:%S', localtime()))
os.makedirs(outputPath)
checkpointer = ModelCheckpoint(outputPath + '/weights_improvement.hdf5',
                               monitor='val_loss',
                               verbose=1,
                               save_best_only=True)
reduce_lr_plateau = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit_generator(generator=dataGenerator.generate(mode='train'),
                              epochs=args.max_epoch,
                              steps_per_epoch=dataGenerator.numTrainBatch,
                              validation_data=dataGenerator.generate(mode='valid'),
                              validation_steps=dataGenerator.numValBatch,
                              callbacks=[checkpointer, reduce_lr_plateau, early_stop])
