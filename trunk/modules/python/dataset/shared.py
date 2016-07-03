import theano as t

def toShared(x, borrow=True, log=None) :
    '''Transfer numpy.ndarry to theano.shared variable.
       NOTE: Shared variables allow for optimized GPU execution

       x      : numpy.ndarray or list object to convert
       borrow : False provides back a deep copy of the memory, True may provide
                a shallow copy only if working on the CPU. GPU memory is always
                deep copied back to the device.
       log    : Logger to use
    '''
    import numpy as np
    if log is not None :
        log.debug('Wrap memory into shared variables')
    return t.shared(np.asarray(x, dtype=t.config.floatX), borrow=borrow)

def fromShared(x, borrow=True, log=None) :
    '''Transfer the memory back to a numpy.ndarry.
       x      : Theano.shared variable to convert
       borrow : False provides back a deep copy of the memory, True may provide
                a shallow copy only if working on the CPU. GPU memory is always
                deep copied back to the host.
       log    : Logger to use
    '''
    if log is not None :
        log.debug('Transfer shared memory back into numpy equivalent')
    return x.get_value(borrow)

def splitToShared(x, borrow=True, castLabelInt=True, log=None) :
    '''Create shared variables for both the input and expectedOutcome vectors.
       x           : This can be a vector list of inputs and expectedOutputs.
                     It is assumed they are of the same length.
                     Format - (data, label)
       borrow      : Should the theano vector use the same buffer as numpy
                     NOTE: GPU execution create new buffer regardless of the
                           value used here.
       castLabelInt: Should the label be casted into int32
       return      : Shared Variable equivalents for these items
                     Format - (data, label)
    '''
    data, label = x
    data = toShared(data, borrow, log)
    label = toShared(label, borrow, log)
    if castLabelInt :
        label = t.tensor.cast(label, 'int32')
    return data, label
