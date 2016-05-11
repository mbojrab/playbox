def calcNumBatches(numElems, batchSize) :
    '''Calculate the number of batches needed to create.'''
    import math
    return int(math.floor(float(numElems) / float(batchSize)))

def createMiniBatch (x, batchSize) :
    '''Resize the tensor to prepare for batched learning.'''
    import numpy as np
    batchShape = tuple([calcNumBatches(len(x), batchSize), batchSize] +
                       list(x.shape[1:]))
    return np.resize(x, batchShape)