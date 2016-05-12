import numpy as np

def calcNumBatches(numElems, batchSize) :
    '''Calculate the number of batches needed to create.'''
    import math
    return int(math.floor(float(numElems) / float(batchSize)))

def resizeMiniBatch (x, batchSize) :
    '''Resize the tensor to prepare for batched learning.'''
    batchShape = tuple([calcNumBatches(len(x), batchSize), batchSize] +
                       list(x.shape[1:]))
    return np.resize(x, batchShape)