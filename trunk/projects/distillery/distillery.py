﻿import argparse, os
import numpy as np
from builder.args import addLoggingParams, addEarlyStop, addSupDataParams
from builder.profiler import setupLogging

from nn.contiguousLayer import ContiguousLayer
from nn.convolutionalLayer import ConvolutionalLayer
from dataset.ingest.labeled import ingestImagery
from nn.trainUtils import trainSupervised

from distill.net import DistilleryTrainer

def createNetwork(inputSize, numKernels, numNeurons, numLabels) :
    from nn.net import ClassifierNetwork

    localPath = './local.pkl.gz'
    network = ClassifierNetwork()

    lr = [.08, .05, .02]
    mr = [.8, .8, .8]

    # add convolutional layers
    network.addLayer(ConvolutionalLayer(
        layerID='c1', inputSize=inputSize,
        kernelSize=(numKernels,inputSize[1],3,3),
        downsampleFactor=(3,3), randomNumGen=rng,
        learningRate=lr[0], momentumRate=mr[0]))
    # add fully connected layers
    network.addLayer(ContiguousLayer(
        layerID='f2', 
        inputSize=(network.getNetworkOutputSize()[0],
                   np.prod(network.getNetworkOutputSize()[1:])),
        numNeurons=numNeurons, randomNumGen=rng,
        learningRate=lr[1], momentumRate=mr[1]))
    network.addLayer(ContiguousLayer(
        layerID='f3', inputSize=network.getNetworkOutputSize(), 
        numNeurons=numLabels, learningRate=lr[2], momentumRate=mr[2], 
        activation=None, randomNumGen=rng))

    # save it to disk in order to load it into both networks
    network.save(localPath)

    return localPath


'''This application will distill a deep network into a shallow one.'''
if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    addEarlyStop(parser)
    addLoggingParams(parser)
    parser.add_argument('--kernel', dest='kernel', type=int, default=6,
                        help='Number of Convolutional Kernels in each Layer.')
    parser.add_argument('--neuron', dest='neuron', type=int, default=120,
                        help='Number of Neurons in Hidden Layer.')
    parser.add_argument('--softness', dest='softness', type=float, default=4.0,
                        help='Softness factor in softmax function.')
    parser.add_argument('--factor', dest='factor', type=float, default=0.8,
                        help='Factor of error coming from deep transfer.')
    parser.add_argument('--deep', dest='deep', type=str, default=None,
                        help='Synapse for the deep network to distill. This ' +
                        'network should be trained and ready.')
    addSupDataParams(parser, 'distillery')
    options = parser.parse_args()

    # setup the logger
    log, prof = setupLogging(options, 'distillery')

    # create a random number generator for efficiency
    from numpy.random import RandomState
    from operator import mul
    from time import time
    rng = RandomState(int(time()))
    #rng = RandomState(4567) # always initialize the same

    # NOTE: The pickleDataset will silently use previously created pickles if
    #       one exists (for efficiency). So watch out for stale pickles!
    # NOTE: User may pass a dark pickle into here, and the logic will react
    #       appropriately to the situation.
    train, test, labels = ingestImagery(filepath=options.data, shared=True,
                                        batchSize=options.batchSize,
                                        holdoutPercentage=options.holdout,
                                        log=log)
    inputSize = train[0].shape.eval()

    # create a file with pre-initialized weights so both networks use the same
    # baseline for testing.
    if options.synapse is None :
        networkFile = createNetwork(inputSize=inputSize[1:],
                                    numKernels=options.kernel,
                                    numNeurons=options.neuron,
                                    numLabels=labels.shape[0])
    else :
        networkFile = options.synapse

    regScale = 1. / (2 * options.kernel * 5 * 5 + 
                     options.neuron + labels.shape[0])

    # load the shallow network
    distNet = DistilleryTrainer(train, test, labels, regType='L2',
                                regScaleFactor=regScale,
                                softmaxTemp=options.softness,
                                transFactor=options.factor,
                                filepath=networkFile, prof=prof)
    # user has not specified a dark pickle infused with additional knowledge
    # from a deep network. In this case, we had the deep network directly to
    # the object in order to get the soft targets JIT
    if len(train) == 2 :
        distNet.loadDeepNetwork(options.deep)

    # perform distilled training
    distFile = trainSupervised(distNet, __file__, options.data, 
                               numEpochs=options.limit, stop=options.stop, 
                               base=options.base + '_distilled', log=log)

    # cleanup the area
    if options.synapse is None :
        os.remove(networkFile)
