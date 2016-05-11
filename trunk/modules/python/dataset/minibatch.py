import numpy as np

def deinterleaveTuples(x) :
    '''Disentangles the tuple, such that each is contigous in a list.
       x      : list of tuples to deinterleave. All objects are assumed to be
                arrays, which will be combined into a single tensor
       return : tuple of tensors. The length of the tuple will equal that of
                the individual members in the input
    '''
    numTuples = len(x)
    if numTuples == 0 :
        raise Exception('Cannot deinterleave an empty list.')
    numElems = len(x[0])

    # extract each item into contiguous memory tensor
    ret = []
    temp = np.concatenate(x)
    for ii in range(numElems) :
        ret.append(np.resize(np.concatenate(temp[ii::numElems]),
                             tuple([numTuples] + list(temp[0][ii].shape))))
    return ret

def calcNumBatches(numElems, batchSize) :
    '''Calculate the number of batches needed to create.'''
    import math
    return int(math.floor(float(numElems) / float(batchSize)))

def resizeMiniBatch (x, batchSize) :
    '''Resize the tensor to prepare for batched learning.'''
    batchShape = tuple([calcNumBatches(len(x), batchSize), batchSize] +
                       list(x.shape[1:]))
    return np.resize(x, batchShape)