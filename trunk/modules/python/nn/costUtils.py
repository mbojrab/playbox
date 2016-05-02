import theano.tensor as t

def crossEntropyLoss (p, q, axis=None):
    ''' for these purposes this is equivalent to Negative Log Likelihood
        this is the average of all cross-entropies in our guess
        p    - the target value
        q    - the current estimate
        axis - the axis in which to sum across -- used for multi-dimensional
    '''
    return t.mean(t.sum(t.nnet.binary_crossentropy(q, p), axis=axis))

def meanSquaredLoss (p, q) :
    ''' for these purposes this is equivalent to Negative Log Likelihood
        p    - the target value
        q    - the current estimate
    '''
    return t.mean((q - p) ** 2)

def leastAbsoluteDeviation(a, batchSize=None, scaleFactor=1.) :
    '''L1-norm provides 'Least Absolute Deviation' --
       built for sparse outputs and is resistent to outliers

       a           - input matrix
       batchSize   - number of inputs in the batchs
       scaleFactor - scale factor for the regularization
    '''
    if batchSize is None :
        return t.mean(t.sum(t.abs_(a)) // batchSize) * scaleFactor
    else :
        return t.sum(t.abs_(a)) * scaleFactor

def leastSquares(a, batchSize=None, scaleFactor=1.) :
    '''L2-norm provides 'Least Squares' --
       built for dense outputs and is computationally stable at small errors

       a           - input matrix
       batchSize   - number of inputs in the batchs
       scaleFactor - scale factor for the regularization

       NOTE: a decent scale factor may be the 1. / numNeurons
    '''
    if batchSize is None :
        return t.mean(t.sum(a ** 2) // batchSize) * scaleFactor
    else :
        return t.mean(t.sum(t.abs_(a))) * scaleFactor

def computeJacobian(a, wrt, batchSize, inputSize, numNeurons) :
    ''' compute a jacobian for the matrix 'out' with respect to 'wrt'.

        This is the first order partials of the output with respect to the 
        weights. This produces a matrix the same size as the input that
        produced the output vector.

        a          - the output matrix for the layer (batchSize, numNeurons)
        wrt        - matrix used to generate 'mat'. This is usually the weight
                     matrix. (inputSize, numNeurons)
        batchSize  - number of inputs in the batch
        inputSize  - size of each input
        numNeurons - number of neurons in the weight matrix
        return     - (batchSize, inputSize)
    '''
    aReshape = (batchSize, 1, numNeurons)
    wrtReshape = (1, inputSize, numNeurons)
    return t.reshape(a * (1 - a), aReshape) * t.reshape(wrt, wrtReshape)
