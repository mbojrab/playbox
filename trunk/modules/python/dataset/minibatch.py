import numpy as np

def calcNumBatches(numElems, batchSize) :
    '''Calculate the number of batches needed to create.'''
    import math
    return int(math.floor(float(numElems) / float(batchSize)))

def resizeMiniBatch (x, batchSize, log=None) :
    '''Resize the tensor to prepare for batched learning.'''
    if log is not None :
        log.debug('Resizing memory to batchSize of [' + str(batchSize) + ']')

    batchShape = tuple([calcNumBatches(len(x), batchSize), batchSize] +
                       list(x.shape[1:]))
    return np.resize(x, batchShape)