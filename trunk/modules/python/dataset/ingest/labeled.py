import numpy as np
import theano.tensor as t

def ingestImagery(filepath, shared=True, holdoutPercentage=.05, minTest=5,
                  batchSize=1, log=None) :
    '''Load the labeled dataset into memory. This is formatted such that the
       directory structure becomes the labels, and all imagery within the 
       directory will be assigned this label. All images in any directory is
       required to have the same dimensions.

       filepath          : This can be a cPickle, a path to the directory
                           structure.
       shared            : Load data into shared variables for training --
                           NOTE: this is only a user suggestion. However the
                                 size of the data will ultimately determine
                                 how its loaded.
       holdoutPercentage : Percentage of the data to holdout for testing
       minTest           : Hard minimum on holdout if percentage is low
       batchSize         : Size of a mini-batch
       log               : Logger for tracking the progress
       return :
           Format -- 
           (trainData, trainLabel), (testData, testLabel), labels

           The 'trainLabel' and 'testLabel' are integer values corresponding
           to the index into the 'labels' string vector. this provides a
           better means to identify errors during back propagation, but
           still allows label finding during classification.

           TODO: Consider returning these as objects for more intuitive
                 indexing. For now numpy indexing is sufficient.
    '''
    from dataset.ingest.preprocHDF5 import reuseableIngest, \
                                           checkAvailableMemory
    from dataset.shared import splitToShared

    # Load the dataset to memory
    train, test, labels = reuseableIngest(filepath=filepath,
                                          holdoutPercentage=holdoutPercentage,
                                          minTest=minTest,
                                          batchSize=batchSize,
                                          saveLabels=True,
                                          log=log)

    # verify it has labels
    if train[1] is None or test[1] is None or labels is None :
        raise ValueError('Unlabeled HDF5 cannot be use for labeled ' +
                         'processing [' + filepath + ']')

    # calculate the memory needed by this dataset
    floatsize = float(np.dtype(t.config.floatX).itemsize)
    intsize = float(np.dtype(np.int32).itemsize)
    dt = [floatsize, intsize, floatsize, intsize]
    dataMemoryConsumption = \
        np.prod(np.asarray(train[0].shape, dtype=np.float32)) * dt[0] + \
        np.prod(np.asarray(train[1].shape, dtype=np.float32)) * dt[1] + \
        np.prod(np.asarray(test[0].shape,  dtype=np.float32)) * dt[2] + \
        np.prod(np.asarray(test[1].shape,  dtype=np.float32)) * dt[3]

    # check physical memory constraints
    shared = checkAvailableMemory(dataMemoryConsumption, shared, log)

    # load each into shared variables -- 
    # this avoids having to copy the data to the GPU between each call
    if shared is True :
        if log is not None :
            log.debug('Transfer the memory into shared variables')
        try :
            tr = splitToShared(train)
            te = splitToShared(test)
            return tr, te, labels
        except :
            pass
    return train, test, labels
