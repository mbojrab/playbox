import numpy as np

def calcNumBatches(numElems, batchSize) :
    '''Calculate the number of batches needed to create.'''
    import math
    return int(math.floor(float(numElems) / float(batchSize)))

def resizeMiniBatch (x, batchSize, log=None) :
    '''Resize the tensor to prepare for batched learning.'''
    if log is not None :
        log.debug('Resizing memory to batchSize of [' + str(batchSize) + ']')

    # the deinterleave confuses numpy as to the real dimensions of the tensor
    # now we concatenate the object back into a single unit and account for
    # numpy removing dimensions.
    if x.shape[0] > 0 and isinstance(x[0], np.ndarray) :
        x = np.reshape(np.concatenate(x), [x.shape[0]] + list(x[0].shape))
    if x.ndim == 1 :
        batchShape = [calcNumBatches(x.shape[0], batchSize), batchSize]
    else :
        batchShape = [calcNumBatches(x.shape[0], batchSize), batchSize] + \
                     list(x.shape[1:])
    return np.resize(x, batchShape)

def makeContiguous(x, batchSize=1, log=None) :
    '''Disentangles the tuple, such that each is contigous in a list.
       x      : list of tuples to deinterleave. All objects are assumed to be
                arrays, which will be combined into a single tensor
       return : tuple of tensors. The length of the tuple will equal that of
                the individual members in the input
    '''
    import numpy as np

    if log is not None :
        log.debug('Making buffers contiguous in memory')

    numTuples = len(x)
    if numTuples == 0 :
        raise Exception('Cannot deinterleave an empty list.')
    numElems = len(x[0])

    # extract each item into contiguous memory tensor
    temp = np.concatenate(x)
    ret = [resizeMiniBatch(temp[ii::numElems], batchSize) \
           for ii in range(numElems)]
    return tuple(ret)