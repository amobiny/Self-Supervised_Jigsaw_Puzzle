import h5py
import numpy as np

data_path = '/home/cougarnet.uh.edu/amobiny/Desktop/DATASETS/Lung_Nodule_data/Lung_Nodule_2d.h5'

h5f = h5py.File(data_path, 'r')
train_img = h5f['X_train'][:]
train_label = h5f['Y_train'][:]
val_img = h5f['X_valid'][:]
val_label = h5f['Y_valid'][:]
h5f.close()

train_mean = np.mean(train_img, axis=0)
train_std = np.std(train_img, axis=0)

h5f = h5py.File('Lung_Nodule_2d.h5', 'w')
h5f.create_dataset('train_img', data=train_img)
h5f.create_dataset('train_label', data=train_label)
h5f.create_dataset('val_img', data=val_img)
h5f.create_dataset('val_label', data=val_label)
h5f.create_dataset('train_mean', data=train_mean)
h5f.create_dataset('train_std', data=train_std)
h5f.close()
