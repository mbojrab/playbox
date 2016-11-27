import theano as t


def isShared(x) :
    '''Test if the data sent is in a Theano shared variable. It is treated
       differently if this check is true.
    '''
    typVar = str(type(x))
    return 'SharedVariable' in typVar or 'TensorVariable' in typVar

def getShape(x) :
    '''Grab the shape in an appropriate manner.'''
    return x.shape.eval() if isShared(x) else x.shape

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
    data, label = x[:2]

    # transfer the data
    data = toShared(data, borrow, log)

    # transfer the hard labels
    label = toShared(label, borrow, log)
    if castLabelInt :
        # NOTE: this resets the value back to TensorVariable
        label = t.tensor.cast(label, 'int32')
    ret = [data, label]

    # exchange any remaining entries
    if len(x) > 2 :
        ret.extend([toShared(d, borrow, log) for d in x[2:]])
    return ret
