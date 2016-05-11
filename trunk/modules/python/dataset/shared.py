import theano as t

def loadShared(x, borrow=True) :
    '''Transfer numpy.array to theano.shared variable.
       NOTE: Shared variables allow for optimized GPU execution
    '''
    import numpy as np
    if not isinstance(x, np.ndarray) :
        x = np.asarray(x, dtype=t.config.floatX)
    return t.shared(x, borrow=borrow)

def splitToShared(x, borrow=True) :
    '''Create shared variables for both the input and expectedOutcome vectors.
       x      : This can be a vector list of inputs and expectedOutputs. It is
                assumed they are of the same length.
                    Format - (data, label)
       borrow : Should the theano vector accept responsibility for the memory
       return : Shared Variable equivalents for these items
                    Format - (data, label)
    '''
    data, label = x
    return (loadShared(data), t.tensor.cast(loadShared(label), 'int32'))
