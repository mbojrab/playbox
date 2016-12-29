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
       return : (trainData, testData)
    '''
    from dataset.ingest.preprocHDF5 import reuseableIngest
    from dataset.shared import toShared

    # Load the dataset to memory
    train, test, labels = reuseableIngest(filepath=filepath,
                                          shared=shared,
                                          holdoutPercentage=holdoutPercentage,
                                          minTest=minTest,
                                          batchSize=batchSize,
                                          saveLabels=False,
                                          log=log)
    train, test = train[0], test[0]

    # calculate the memory needed by this dataset
    floatsize = float(np.dtype(t.config.floatX).itemsize)
    intsize = float(np.dtype(np.int32).itemsize)
    dt = [floatsize, intsize, floatsize, intsize]
    dataMemoryConsumption = \
        np.prod(np.asarray(train.shape, dtype=np.float32)) * floatsize + \
        np.prod(np.asarray(test.shape,  dtype=np.float32)) * floatsize

    # check physical memory constraints
    shared = checkAvailableMemory(dataMemoryConsumption, shared, log)

    # load each into shared variables -- 
    # this avoids having to copy the data to the GPU between each call
    if shared is True :
        if log is not None :
            log.debug('Transfer the memory into shared variables')
        try :
            tr = toShared(train, log=log)
            te = toShared(test, log=log)
            return tr, te
        except :
            pass
    return train, test
