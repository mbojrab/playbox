
def convertActivation(op) :
    '''This operation converts theano activation function to and from string
       format. This is used for improved pickle compression.
    '''
    import theano
    if op is None :
        return op

    # create a map to perform the conversion
    opMap = {'tanh' : theano.tensor.tanh, 
             'sigmoid' : theano.tensor.nnet.sigmoid}

    # attempt to perform the lookup
    if op in opMap.keys() :
        return opMap[op]

    # inverse the map and attempt the lookup again
    invOpMap = {v: k for k, v in opMap.items()}
    if op in invOpMap.keys() :
        return invOpMap[op]

    raise ValueError('Operation [' + op + '] is not supported.')
