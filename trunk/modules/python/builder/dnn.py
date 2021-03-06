﻿import theano.tensor as t
from numpy.random import RandomState
from time import time

def __addDefaults (param, default, size) :
    return [default] * size if len(param) == 0 else param

def __verifyLengths (param, ref, paramName, refName) :
    if len(param) != len(ref) :
        raise ValueError('Different number of parameters between [' + 
                         paramName + '] and [' + refName + ']')

def setupCommandLine (base='cnn', multiLoad=False) :
    '''Create a argparser with the proper CNN command line parameters,
       and return the options class.
    '''
    import argparse
    from builder.args import addLoggingParams, \
                             addDebuggingParams, \
                             addEarlyStop, \
                             addSupDataParams, \
                             addSupContiguousParams, \
                             addSupConvolutionalParams

    # setup the common command line options
    parser = argparse.ArgumentParser()
    addLoggingParams(parser)
    addDebuggingParams(parser)
    addEarlyStop(parser)
    addSupDataParams(parser, base, multiLoad=multiLoad)
    addSupConvolutionalParams(parser)
    addSupContiguousParams(parser)

    # parse the user-provided options
    return parser.parse_args()

def addConvolutionalNN (network, inputSize, options, rng=None, prof=None) :
    '''Add ConvolutionalAE layers to the network and return it.'''
    from nn.convolutionalLayer import ConvolutionalLayer

    # create a rng if one was not supplied
    if rng is None :
        rng = RandomState(int(time()))

    # reset if the network already has layers
    if network.getNumLayers() > 0 :
        inputSize = network.getNetworkOutputSize()

    # perform validation and defaulting
    __verifyLengths (options.kernelSize, options.kernel,
                     'kernelSize', 'kernel')
    __verifyLengths (options.downsample, options.kernel,
                     'downsample', 'kernel')
    __verifyLengths (options.learnC, options.kernel,
                     'learnC', 'kernel')
    options.momentumC = __addDefaults(options.momentumC, .0, 
                                      len(options.kernel))
    options.dropoutC = __addDefaults(options.dropoutC, 1.,
                                     len(options.kernel))

    # add each layer contiguously
    for k,ks,do,l,m,dr,sc in zip(options.kernel, options.kernelSize,
                                 options.downsample, options.learnC,
                                 options.momentumC, options.dropoutC) :

        # add a convolutional layer as defined
        network.addLayer(ConvolutionalLayer(
            layerID='conv' + str(network.getNumLayers() + 1), 
            inputSize=inputSize, kernelSize=(k,inputSize[1],ks,ks),
            downsampleFactor=[do,do], dropout=dr, 
            learningRate=l, momentumRate=m,
            activation=t.nnet.sigmoid, randomNumGen=rng))

        # prepare for the next layer
        inputSize = network.getNetworkOutputSize()


def addContiguousNN (network, inputSize, options, rng=None, prof=None) :
    '''Add ContiguousAE layers to the network and return it.'''
    import numpy as np
    from nn.contiguousLayer import ContiguousLayer

    # create a rng if one was not supplied
    if rng is None :
        rng = RandomState(int(time()))

    # reset if the network already has layers
    if network.getNumLayers() > 0 :
        inputSize = network.getNetworkOutputSize()

    # flatten the input size -- just in case
    inputSize = (inputSize[0], np.prod(inputSize[1:]))

    # perform validation and defaulting
    __verifyLengths (options.learnF, options.neuron, 'learnF', 'neuron')
    options.momentumF = __addDefaults(options.momentumF, .0,
                                      len(options.neuron))
    options.dropoutF = __addDefaults(options.dropoutF, 1.,
                                     len(options.neuron))

    # add each layer contiguously
    for n,l,m,dr,sc in zip(options.neuron, options.learnF,
                           options.momentumF, options.dropoutF) :

        # add a fully-connected layer as defined
        network.addLayer(ContiguousAutoEncoder(
            layerID='fully' + str(network.getNumLayers() + 1),
            inputSize=inputSize, numNeurons=n, learningRate=l,
            activation=t.nnet.sigmoid, dropout=dr,
            momentumRate=m, randomNumGen=rng))

        # prepare for the next layer
        inputSize = network.getNetworkOutputSize()


def buildNetwork(network, inputSize, options, rng=None, prof=None) :
    '''Build the Stacked AutoEncoder network in an automated way.'''

    # use the same random number generator across all layers for efficiency
    if rng is None :
        rng = RandomState(int(time()))

    if prof is not None :
        prof.startProfile('Initialize the Network', 'info')

    # add any user-defined ConvolutionalAE layers first
    addConvolutionalNN (network, inputSize, options, rng=rng, prof=prof)

    # add any user-defined ContiguousAE layers second
    addContiguousNN (network, inputSize, options, rng=rng, prof=prof)

    if prof is not None :
        prof.endProfile()

    return network
