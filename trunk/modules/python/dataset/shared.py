import theano as t

def loadShared(x, borrow=True, log=None) :
    '''Transfer numpy.array to theano.shared variable.
       NOTE: Shared variables allow for optimized GPU execution
    '''
    import numpy as np
    if log is not None :
        log.debug('Wrap memory into shared variables')
    return t.shared(np.asarray(x, dtype=t.config.floatX), borrow=borrow)

def splitToShared(x, borrow=True, castInt=True, log=None) :
    '''Create shared variables for both the input and expectedOutcome vectors.
       x      : This can be a vector list of inputs and expectedOutputs. It is
                assumed they are of the same length.
                    Format - (data, label)
       borrow : Should the theano vector use the same buffer as numpy
       castInt: Should the label be casted into int32
       return : Shared Variable equivalents for these items
                    Format - (data, label)
    '''
    data, label = x
    data = loadShared(data, borrow, log)
    label = loadShared(label, borrow, log)
    if castInt :
        label = t.tensor.cast(label, 'int32')
    return data, label
