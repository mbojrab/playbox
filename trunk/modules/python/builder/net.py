import theano.tensor as t
from nn.net import ClassifierNetwork
from ae.encoder import AutoEncoder
import numpy as np

def buildSAENetwork(network, layerInputSize, regType, regValue,
                    kernelConv, kernelSizeConv, downsampleConv, 
                    learnConv, momentumConv, dropoutConv, sparseConv,
                    neuronFull, learnFull, momentumFull, dropoutFull,
                    sparseFull, prof=None) :
    '''Build the Stacked AutoEncoder network in an automated way.'''
    from ae.contiguousAE import ContiguousAutoEncoder
    from ae.convolutionalAE import ConvolutionalAutoEncoder

    from operator import mul
    from numpy.random import RandomState
    from time import time
    rng = RandomState(int(time()))

    if prof is not None :
        prof.startProfile('Initialize the Network', 'info')

    # prepare for the next layer
    def prepare(network, count) :
        return (count + 1, 
                network.getNetworkOutputSize())

    layerCount = 1
    if kernelConv is not None :
        for k,ks,do,l,m,dr,sc in zip(kernelConv, kernelSizeConv, 
                                     downsampleConv, learnConv, momentumConv,
                                     dropoutConv, sparseConv) :
            # add a convolutional layer as defined
            network.addLayer(ConvolutionalAutoEncoder(
                layerID='conv' + str(layerCount), 
                regType=regType, contractionRate=regValue,
                inputSize=layerInputSize,
                kernelSize=(k,layerInputSize[1],ks,ks),
                downsampleFactor=[do,do], dropout=dr, 
                learningRate=l, forceSparsity=sc, momentumRate=m,
                activation=t.nnet.sigmoid, randomNumGen=rng))

            # prepare for the next layer
            layerCount, layerInputSize = prepare(network, layerCount)

    # add reset in case user uses removeLayer() logic
    if network.getNumLayers() > 0 :
        layerInputSize = network.getNetworkOutputSize()

    # update to transition for fully connected layers
    layerInputSize = (layerInputSize[0], np.prod(layerInputSize[1:]))
    for n,l,m,dr,sc in zip(neuronFull, learnFull, momentumFull,
                           dropoutFull, sparseFull) :
        # add a fully-connected layer as defined
        network.addLayer(ContiguousAutoEncoder(
            layerID='fully' + str(layerCount),
            regType=regType, contractionRate=regValue,
            inputSize=layerInputSize, numNeurons=n, learningRate=l,
            activation=t.nnet.sigmoid, dropout=dr, forceSparsity=sc,
            momentumRate=m, randomNumGen=rng))

        # prepare for the next layer
        layerCount, layerInputSize = prepare(network, layerCount)

    if prof is not None :
        prof.endProfile()

    return network
