import h5py
import tensorflow as tf
from hamming_set.generate_hamming_set import hamming_set
from config import args
import numpy as np
import os
from models.AlexNet.Siamese import Siamese_AlexNet
from models.CapsNet.Siamese import Siamese_CapsNet


def main(_):
    if args.generateHammingSet:
        hamming_set(args.numCrops, args.hammingSetSize,
                    args.selectionMethod, args.hammingFilePath)

    h5f = h5py.File('./hamming_set/' + args.hammingFileName, 'r')
    HammingSet = np.array(h5f['max_hamming_set'])
    h5f.close()
    if args.mode not in ['train', 'test', 'predict']:
        print('invalid mode: ', args.mode)
        print("Please input a mode: train, test, or predict")
    else:
        model = Siamese_AlexNet(tf.Session(), args, HammingSet)
        if not os.path.exists(args.modeldir+args.run_name):
            os.makedirs(args.modeldir+args.run_name)
        if not os.path.exists(args.logdir+args.run_name):
            os.makedirs(args.logdir+args.run_name)
        if not os.path.exists(args.savedir+args.run_name):
            os.makedirs(args.savedir+args.run_name)
        if args.mode == 'train':
            model.train()
        elif args.mode == 'test':
            model.test(epoch_num=6)


if __name__ == '__main__':
    # configure which gpu or cpu to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
    tf.app.run()
