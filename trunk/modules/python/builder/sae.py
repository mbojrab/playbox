import theano.tensor as t
from numpy.random import RandomState
from time import time

def __addDefaults (param, default, size) :
    return [default] * size if len(param) == 0 else param

def __verifyLengths (param, ref, paramName, refName) :
    if len(param) != len(ref) :
        raise ValueError('Different number of parameters between [' + 
                         paramName + '] and [' + refName + ']')

def addConvolutionalAE (network, inputSize, kernel, kernelSize, downsample, 
                        learnRate, regType='L2', regValue=.0001,
                        momentumRate=[], dropout=[], forceSparsity=[],
                        rng=None, prof=None) :
    '''Add ConvolutionalAE layers to the network and return it.'''
    from ae.convolutionalAE import ConvolutionalAutoEncoder

    # create a rng if one was not supplied
    if rng is None :
        rng = RandomState(int(time()))

    # reset if the network already has layers
    if network.getNumLayers() > 0 :
        inputSize = network.getNetworkOutputSize()

    # perform validation and defaulting
    __verifyLengths (kernelSize, kernel, 'kernelSize', 'kernel')
    __verifyLengths (downsample, kernel, 'downsample', 'kernel')
    __verifyLengths (learnRate, kernel, 'learnRate', 'kernel')
    momentumRate = __addDefaults(momentumRate, .0, len(kernel))
    dropout = __addDefaults(dropout, 1., len(kernel))
    forceSparsity = __addDefaults(forceSparsity, True, len(kernel))

    # add each layer contiguously
    for k,ks,do,l,m,dr,sc in zip(kernel, kernelSize, downsample,
                                 learnRate, momentumRate,
                                 dropout, forceSparsity) :

        # add a convolutional layer as defined
        network.addLayer(ConvolutionalAutoEncoder(
            layerID='conv' + str(network.getNumLayers() + 1), 
            regType=regType, contractionRate=regValue,
            inputSize=inputSize,
            kernelSize=(k,inputSize[1],ks,ks),
            downsampleFactor=[do,do], dropout=dr, 
            learningRate=l, forceSparsity=sc, momentumRate=m,
            activation=t.nnet.sigmoid, randomNumGen=rng))

        # prepare for the next layer
        inputSize = network.getNetworkOutputSize()


def addContiguousAE (network, inputSize, neuron, learnRate, 
                     regType='L2', regValue=.0001, momentumRate=[],
                     dropout=[], forceSparsity=[], rng=None, prof=None) :
    '''Add ContiguousAE layers to the network and return it.'''
    import numpy as np
    from ae.contiguousAE import ContiguousAutoEncoder

    # create a rng if one was not supplied
    if rng is None :
        rng = RandomState(int(time()))

    # reset if the network already has layers
    if network.getNumLayers() > 0 :
        inputSize = network.getNetworkOutputSize()

    # flatten the input size -- just in case
    inputSize = (inputSize[0], np.prod(inputSize[1:]))

    # perform validation and defaulting
    __verifyLengths (learnRate, neuron, 'learnRate', 'neuron')
    momentumRate = __addDefaults(momentumRate, .0, len(neuron))
    dropout = __addDefaults(dropout, 1., len(neuron))
    forceSparsity = __addDefaults(forceSparsity, True, len(neuron))

    # add each layer contiguously
    for n,l,m,dr,sc in zip(neuron, learnRate, momentumRate,
                           dropout, forceSparsity) :

        # add a fully-connected layer as defined
        network.addLayer(ContiguousAutoEncoder(
            layerID='fully' + str(network.getNumLayers() + 1),
            regType=regType, contractionRate=regValue,
            inputSize=inputSize, numNeurons=n, learningRate=l,
            activation=t.nnet.sigmoid, dropout=dr, forceSparsity=sc,
            momentumRate=m, randomNumGen=rng))

        # prepare for the next layer
        inputSize = network.getNetworkOutputSize()


def buildSAENetwork(network, inputSize, regType, regValue,
                    kernelConv, kernelSizeConv, downsampleConv, 
                    learnConv, momentumConv, dropoutConv, sparseConv,
                    neuronFull, learnFull, momentumFull, dropoutFull,
                    sparseFull, rng=None, prof=None) :
    '''Build the Stacked AutoEncoder network in an automated way.'''
    from ae.contiguousAE import ContiguousAutoEncoder

    # use the same random number generator across all layers for efficiency
    if rng is None :
        rng = RandomState(int(time()))

    if prof is not None :
        prof.startProfile('Initialize the Network', 'info')

    # add any user-defined ConvolutionalAE layers first
    addConvolutionalAE (network, inputSize, kernelConv, kernelSizeConv,
                        downsampleConv, learnConv, regType, regValue, 
                        momentumConv, dropoutConv, sparseConv, 
                        rng=rng, prof=prof)

    # add any user-defined ContiguousAE layers second
    addContiguousAE (network, inputSize, neuronFull, learnFull, 
                     regType, regValue, momentumFull, dropoutFull, 
                     sparseFull, rng=rng, prof=prof)

    if prof is not None :
        prof.endProfile()

    return network
