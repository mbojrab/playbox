'''From Keras library'''
from cifar import load_batch
import numpy as np
import os
from six.moves import cPickle


def load_data(label_mode='fine'):
    if label_mode not in ['fine', 'coarse']:
        raise Exception('label_mode must be one of "fine" "coarse".')

    dirname = './cifar-100-python'

    nb_test_samples = 10000
    nb_train_samples = 50000

    fpath = os.path.join(dirname, 'train')
    X_train, y_train = load_batch(fpath, label_key=label_mode+'_labels')

    fpath = os.path.join(dirname, 'test')
    X_test, y_test = load_batch(fpath, label_key=label_mode+'_labels')

    #y_train = np.reshape(y_train, (len(y_train), 1))
    #y_test = np.reshape(y_test, (len(y_test), 1))

    labelNames = []
    fpath = os.path.join(dirname, 'meta')
    with open(fpath, 'rb') as f :
        labelNames = cPickle.load(f)

    return (X_train, y_train), (X_test, y_test), labelNames
