import theano.tensor as t
import theano
from nn.net import ClassifierNetwork
from ae.encoder import AutoEncoder
import numpy as np

def buildSAENetwork(network, inputSize,
                    kernelConv, kernelSizeConv, downsampleConv,
                    learnConv, momentumConv, dropoutConv,
                    neuronFull, learnFull, momentumFull, dropoutFull, 
                    log=None) :
    '''Build the network in an automated way.'''
    from ae.convolutionalAE import ConvolutionalAutoEncoder
    from ae.contiguousAE import ContiguousAutoEncoder
    from six.moves import reduce
    from operator import mul
    from numpy.random import RandomState
    from time import time
    rng = RandomState(int(time()))

    if log is not None :
        log.info('Initialize the Network')

    def prepare(network, count) :
        return (count + 1, network.getNetworkOutputSize())

    layerCount = 1
    layerInputSize = inputSize
    for k,ks,do,l,m,dr in zip(kernelConv, kernelSizeConv, downsampleConv, 
                              learnConv, momentumConv, dropoutConv) :
        # add a convolutional layer as defined
        network.addLayer(ConvolutionalAutoEncoder(
            layerID='conv' + str(layerCount), 
            inputSize=layerInputSize, kernelSize=(k,layerInputSize[1],ks,ks),
            downsampleFactor=[do,do], dropout=dr, learningRate=l,
            randomNumGen=rng))

        # prepare for the next layer
        layerCount, layerInputSize = prepare(network, layerCount)

    # update to transition for fully connected layers
    layerInputSize = (layerInputSize[0], reduce(mul, layerInputSize[1:]))
    for n,l,m,dr in zip(neuronFull, learnFull, momentumFull, dropoutFull) :
        # add a fully-connected layer as defined
        network.addLayer(ContiguousAutoEncoder(
            layerID='fully' + str(layerCount), 
            inputSize=layerInputSize, numNeurons=n, learningRate=l,
            dropout=dr, randomNumGen=rng))            

        # prepare for the next layer
        layerCount, layerInputSize = prepare(network, layerCount)

    return network
