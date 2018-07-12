from keras.layers import Dense, Dropout, Concatenate, Input

from DataLoader.DataGenerator import DataGenerator
from models.Residual_Network import ResNet34
from config import args


def contextFreeNetwork(args):
    """
    Implemented non-siamese
    tileSize - The dimensions of the jigsaw input
    numPuzzles - the number of jigsaw puzzles

    returns a keras model
    """
    inputShape = (args.tileSize, args.tileSize, 3)
    modelInputs = [Input(inputShape) for _ in range(args.numPuzzles)]
    sharedLayer = ResNet34(inputShape)
    sharedLayers = [sharedLayer(inputTensor) for inputTensor in modelInputs]
    x = Concatenate()(sharedLayers)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(args.hammingSetSize, activation='softmax')(x)
    return x


HammingSet = []
dataGenerator = DataGenerator(args, HammingSet)
