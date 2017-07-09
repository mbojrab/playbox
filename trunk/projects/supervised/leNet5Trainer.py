import argparse
import numpy as np
import theano.tensor as t
from time import time
from numpy.random import RandomState

from nn.net import TrainerNetwork as Net
from nn.contiguousLayer import ContiguousLayer
from nn.convolutionalLayer import ConvolutionalLayer
from dataset.ingest.labeled import ingestImagery
from builder.args import addLoggingParams, addDebuggingParams, \
                         addSupDataParams, addEarlyStop
from builder.profiler import setupLogging
from nn.trainUtils import trainSupervised
from dataset.shared import getShape

'''This is a simple network in the topology of leNet5 the well-known
   MNIST dataset trainer from Yann LeCun. This is capable of training other
   datasets, however the sizing must be correct.
'''
if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    addLoggingParams(parser)
    addDebuggingParams(parser)
    parser.add_argument('--learnC', dest='learnC', type=float, default=.031,
                        help='Rate of learning on Convolutional Layers.')
    parser.add_argument('--learnF', dest='learnF', type=float, default=.015,
                        help='Rate of learning on Fully-Connected Layers.')
    parser.add_argument('--momentum', dest='momentum', type=float, default=.1,
                        help='Momentum rate all layers.')
    parser.add_argument('--dropout', dest='dropout', type=bool, default=False,
                        help='Enable dropout throughout the network. Dropout '\
                             'percentages are based on optimal reported '\
                             'results. NOTE: Networks using dropout need to '\
                             'increase both neural breadth and learning rates')
    parser.add_argument('--kernel', dest='kernel', type=int, default=6,
                        help='Number of Convolutional Kernels in each Layer.')
    parser.add_argument('--neuron', dest='neuron', type=int, default=120,
                        help='Number of Neurons in Hidden Layer.')
    addEarlyStop(parser)
    addSupDataParams(parser, 'leNet5')
    options = parser.parse_args()

    # setup the logger
    log, prof = setupLogging(options, 'cnnTrainer')

    # create a random number generator for efficiency
    rng = RandomState(int(time()))

    # NOTE: The pickleDataset will silently use previously created pickles if
    #       one exists (for efficiency). So watch out for stale pickles!
    train, test, labels = ingestImagery(filepath=options.data, shared=True,
                                        batchSize=options.batchSize,
                                        holdoutPercentage=options.holdout,
                                        log=log)
    trainSize = getShape(train[0])

    # create the network -- LeNet-5
    network = Net(train, test, labels, regType='L2', 
                  regScaleFactor=1. / (2 * options.kernel * 5 * 5 + 
                                       options.neuron + labels.shape[0]),
                  prof=prof, debug=options.debug)

    if options.synapse is not None :
        # load a previously saved network
        network.load(options.synapse)

        # reset the learning rates
        network.setLayerLearningRate(0, options.learnC)
        network.setLayerLearningRate(1, options.learnC)
        network.setLayerLearningRate(2, options.learnF)
        network.setLayerLearningRate(3, options.learnF)

        # reset the momentum ratez
        network.setLayerMomentumRate(0, options.momentum)
        network.setLayerMomentumRate(1, options.momentum)
        network.setLayerMomentumRate(2, options.momentum)
        network.setLayerMomentumRate(3, options.momentum)

    else :
        log.info('Initializing Network...')

        # add convolutional layers
        network.addLayer(ConvolutionalLayer(
            layerID='c1', inputSize=trainSize[1:],
            kernelSize=(options.kernel,trainSize[2],5,5),
            downsampleFactor=(2,2),
            learningRate=options.learnC, momentumRate=options.momentum,
            dropout=.8 if options.dropout else 1.,
            activation=t.nnet.relu, randomNumGen=rng))

        # refactor the output to be (numImages*numKernels, 1, numRows, numCols)
        # this way we don't combine the channels kernels we created in 
        # the first layer and destroy our dimensionality
        network.addLayer(ConvolutionalLayer(
            layerID='c2', inputSize=network.getNetworkOutputSize(), 
            kernelSize=(options.kernel,options.kernel,5,5),
            downsampleFactor=(2,2), 
            learningRate=options.learnC, momentumRate=options.momentum,
            dropout=.5 if options.dropout else 1., 
            activation=t.nnet.relu, randomNumGen=rng))

        # add fully connected layers
        network.addLayer(ContiguousLayer(
            layerID='f3', 
            inputSize=(network.getNetworkOutputSize()[0], 
                       np.prod(network.getNetworkOutputSize()[1:])),
            numNeurons=options.neuron, 
            learningRate=options.learnF, momentumRate=options.momentum,
            dropout=.5 if options.dropout else 1.,
            activation=t.nnet.relu, randomNumGen=rng))
        network.addLayer(ContiguousLayer(
            layerID='f4', inputSize=network.getNetworkOutputSize(), 
            numNeurons=labels.shape[0],
            learningRate=options.learnF, momentumRate=options.momentum,
            activation=t.nnet.relu, randomNumGen=rng))

    trainSupervised(network, __file__, options.data, 
                    numEpochs=options.limit, stop=options.stop, 
                    synapse=options.synapse, base=options.base, 
                    maxEpoch=options.epoch, log=log)
