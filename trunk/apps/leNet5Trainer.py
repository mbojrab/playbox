import argparse
from time import time
from six.moves import reduce

from nn.net import TrainerNetwork as Net
from nn.contiguousLayer import ContiguousLayer
from nn.convolutionalLayer import ConvolutionalLayer
from dataset.ingest.labeled import ingestImagery
from nn.trainUtils import trainSupervised
from nn.profiler import setupLogging, Profiler

'''This is a simple network in the topology of leNet5 the well-known
   MNIST dataset trainer from Yann LeCun. This is capable of training other
   datasets, however the sizing must be correct.
'''
if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    parser.add_argument('--log', dest='logfile', type=str, default=None,
                        help='Specify log output file.')
    parser.add_argument('--level', dest='level', default='INFO', type=str, 
                        help='Log Level.')
    parser.add_argument('--prof', dest='profile', type=str, 
                        default='Application-Profiler.xml',
                        help='Specify profile output file.')
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
    parser.add_argument('--limit', dest='limit', type=int, default=5,
                        help='Number of runs between validation checks.')
    parser.add_argument('--stop', dest='stop', type=int, default=5,
                        help='Number of inferior validation checks to end.')
    parser.add_argument('--holdout', dest='holdout', type=float, default=.05,
                        help='Percent of data to be held out for testing.')
    parser.add_argument('--batch', dest='batchSize', type=int, default=5,
                        help='Batch size for training and test sets.')
    parser.add_argument('--base', dest='base', type=str, default='./leNet5',
                        help='Base name of the network output and temp files.')
    parser.add_argument('--syn', dest='synapse', type=str, default=None,
                        help='Load from a previously saved network.')
    parser.add_argument('data', help='Directory or pkl.gz file for the ' +
                                     'training and test sets')
    options = parser.parse_args()

    # setup the logger
    logName = 'cnnTrainer: ' + options.data
    log = setupLogging(logName, options.level, options.logfile)
    prof = Profiler(log=log, name=logName, profFile=options.profile)

    # create a random number generator for efficiency
    import theano.tensor as t
    from numpy.random import RandomState
    from operator import mul
    rng = RandomState(int(time()))

    # NOTE: The pickleDataset will silently use previously created pickles if
    #       one exists (for efficiency). So watch out for stale pickles!
    train, test, labels = ingestImagery(filepath=options.data, shared=True,
                                        batchSize=options.batchSize,
                                        holdoutPercentage=options.holdout,
                                        log=log)
    trainSize = train[0].shape.eval()

    # create the network -- LeNet-5
    network = Net(train, test, labels, regType='L2', 
                  regScaleFactor=1. / (2 * options.kernel * 5 * 5 + 
                                       options.neuron + len(labels)),
                  prof=prof)

    if options.synapse is not None :
        # load a previously saved network
        network.load(options.synapse)
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
                       reduce(mul, network.getNetworkOutputSize()[1:])),
            numNeurons=options.neuron, 
            learningRate=options.learnF, momentumRate=options.momentum,
            dropout=.5 if options.dropout else 1.,
            activation=t.nnet.relu, randomNumGen=rng))
        network.addLayer(ContiguousLayer(
            layerID='f4', inputSize=network.getNetworkOutputSize(), 
            numNeurons=len(labels),
            learningRate=options.learnF, momentumRate=options.momentum,
            activation=t.nnet.relu, randomNumGen=rng))

    trainSupervised(network, __file__, options.data, 
                    numEpochs=options.limit, stop=options.stop, 
                    synapse=options.synapse, base=options.base, 
                    dropout=options.dropout, learnC=options.learnC, 
                    learnF=options.learnF, momentum=options.momentum, 
                    kernel=options.kernel, neuron=options.neuron, 
                    log=log)
