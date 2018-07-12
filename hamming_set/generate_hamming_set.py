import numpy as np
import itertools
from scipy.spatial.distance import cdist
import h5py


def hamming_set(num_crops, num_permutations, selection, output_file_name):
    """
    generate and save the hamming set
    :param num_crops: number of tiles from each image
    :param num_permutations: Number of permutations to select (i.e. number of classes for the pretext task)
    :param selection: Sample selected per iteration based on hamming distance: [max] highest; [mean] average
    :param output_file_name: name of the output HDF5 file
    """
    P_hat = np.array(list(itertools.permutations(list(range(num_crops)), num_crops)))
    n = P_hat.shape[0]

    for i in range(num_permutations):
        if i == 0:
            j = np.random.randint(n)
            P = np.array(P_hat[j]).reshape([1, -1])
        else:
            P = np.concatenate([P, P_hat[j].reshape([1, -1])], axis=0)

        P_hat = np.delete(P_hat, j, axis=0)
        D = cdist(P, P_hat, metric='hamming').mean(axis=0).flatten()

        if selection == 'max':
            j = D.argmax()
        elif selection == 'mean':
            m = int(D.shape[0] / 2)
            S = D.argsort()
            j = S[np.random.randint(m - 10, m + 10)]

    h5f = h5py.File('./hamming_set/' + output_file_name, 'w')
    h5f.create_dataset(output_file_name, data=P)
    h5f.close()
    print('file created --> ' + output_file_name)


if __name__ == "__main__":
    hamming_set(num_crops=9,
                num_permutations=100,
                selection='max',
                output_file_name='max_hamming_set.h5')
