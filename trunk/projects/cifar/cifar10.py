'''From Keras library'''
from cifar import load_batch
import numpy as np
import os
from six.moves import cPickle

def load_data():
    dirname = './cifar-10-batches-py'
    nb_train_samples = 50000

    X_train = np.zeros((nb_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.zeros((nb_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(dirname, 'data_batch_' + str(i))
        data, labels = load_batch(fpath)
        X_train[(i-1)*10000:i*10000, :, :, :] = data
        y_train[(i-1)*10000:i*10000] = labels

    fpath = os.path.join(dirname, 'test_batch')
    X_test, y_test = load_batch(fpath)

    labelNames = []
    fpath = os.path.join(dirname, 'batches.meta')
    with open(fpath, 'rb') as f :
        labelNames = cPickle.load(f)

    return (X_train, y_train), (X_test, y_test), labelNames
