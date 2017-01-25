import numpy as np
import argparse
from time import time
from builder.args import addLoggingParams, addDebuggingParams, \
                         addEarlyStop, addSupDataParams
from builder.profiler import setupLogging

from ae.net import TrainerSAENetwork
from ae.contiguousAE import ContiguousAutoEncoder
from ae.convolutionalAE import ConvolutionalAutoEncoder
from dataset.ingest.labeled import ingestImagery
from nn.contiguousLayer import ContiguousLayer
from nn.trainUtils import trainUnsupervised, trainSupervised
from nn.net import TrainerNetwork

if __name__ == '__main__' :
    '''This application runs semi-supervised training on a given dataset. The
       utimate goal is to setup the early layers of the Neural Network to
       identify pattern in the data unsupervised learning. Here we used a 
       Stacked Autoencoder (SAE) and greedy training.
       We then translate the SAE to a Neural Network (NN) and add a 
       classification layer. From there we use supervised training to fine-tune
       the weights to classify objects we select as important.
    '''
    parser = argparse.ArgumentParser()
    addDebuggingParams(parser)
    addLoggingParams(parser)
    parser.add_argument('--learnC', dest='learnC', type=float, default=.0031,
                        help='Rate of learning on Convolutional Layers.')
    parser.add_argument('--learnF', dest='learnF', type=float, default=.0015,
                        help='Rate of learning on Fully-Connected Layers.')
    parser.add_argument('--contrF', dest='contrF', type=float, default=None,
                        help='Rate of contraction of the latent space on ' +
                             'Fully-Connected Layers.')
    parser.add_argument('--momentum', dest='momentum', type=float, default=.3,
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
    log, prof = setupLogging(options, 'semiSupervisedTrainer')

    # create a random number generator for efficiency
    from numpy.random import RandomState
    rng = RandomState(int(time()))

    # NOTE: The pickleDataset will silently use previously created pickles if
    #       one exists (for efficiency). So watch out for stale pickles!
    train, test, labels = ingestImagery(filepath=options.data, shared=True,
                                        batchSize=options.batchSize, 
                                        holdoutPercentage=options.holdout, 
                                        log=log)
    trainShape = train[0].shape.eval()

    # create the stacked network -- LeNet-5 (minus the output layer)
    network = TrainerSAENetwork(train, prof=prof, debug=options.debug)

    if options.synapse is not None :
        # load a previously saved network
        network.load(options.synapse)
    else :
        log.info('Initializing Network...')

        # add convolutional layers
        network.addLayer(ConvolutionalAutoEncoder(
            layerID='c1', inputSize=trainShape[1:], 
            kernelSize=(options.kernel,trainShape[2],5,5),
            downsampleFactor=(2,2), randomNumGen=rng,
            dropout=.8 if options.dropout else 1.,
            learningRate=options.learnC))

        # refactor the output to be (numImages*numKernels,1,numRows,numCols)
        # this way we don't combine the channels kernels we created in 
        # the first layer and destroy our dimensionality
        network.addLayer(ConvolutionalAutoEncoder(
            layerID='c2', inputSize=network.getNetworkOutputSize(), 
            kernelSize=(options.kernel,options.kernel,5,5),
            downsampleFactor=(2,2), randomNumGen=rng,
            dropout=.5 if options.dropout else 1.,
            learningRate=options.learnC))

        # add fully connected layers
        network.addLayer(ContiguousAutoEncoder(
            layerID='f3', 
            inputSize=(network.getNetworkOutputSize()[0], 
                       np.prod(network.getNetworkOutputSize()[1:])),
            numNeurons=options.neuron, learningRate=options.learnF,
            dropout=.5 if options.dropout else 1.,
            randomNumGen=rng))

        # the final output layer is removed from the normal NN --
        # the output layer is special, as it makes decisions about
        # patterns identified in previous layers, so it should only
        # be influenced/trained during supervised learning. 

    # train the SAE for unsupervised pattern recognition
    bestNetwork = trainUnsupervised(network, __file__, options.data, 
                                    numEpochs=options.limit, stop=options.stop, 
                                    synapse=options.synapse, base=options.base, 
                                    dropout=options.dropout, 
                                    learnC=options.learnC,
                                    learnF=options.learnF, 
                                    contrF=options.contrF, 
                                    momentum=options.momentum, 
                                    kernel=options.kernel, 
                                    neuron=options.neuron, log=log)

    # translate into a neural network --
    # this transfers our unsupervised pre-training into a decent
    # starting condition for our supervised learning
    network = TrainerNetwork(train, test, labels, filepath=bestNetwork,
                             prof=prof, debug=options.debug)

    # add the classification layer
    network.addLayer(ContiguousLayer(
        layerID='f4', input=network.getNetworkOutput(),
        inputSize=network.getNetworkOutputSize(), numNeurons=labels.shape[0],
        learningRate=options.learnF, randomNumGen=rng))

    # train the NN for supervised classification
    trainSupervised(network, __file__, options.data, 
                    numEpochs=options.limit, stop=options.stop, 
                    synapse=options.synapse, base=options.base, 
                    dropout=options.dropout, learnC=options.learnC, 
                    learnF=options.learnF, momentum=options.momentum, 
                    kernel=options.kernel, neuron=options.neuron, log=log)
