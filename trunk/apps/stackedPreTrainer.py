import argparse
from time import time
from six.moves import reduce

from ae.net import StackedAENetwork
from ae.contiguousAE import ContiguousAutoEncoder
from ae.convolutionalAE import ConvolutionalAutoEncoder
from dataset.ingest.labeled import ingestImagery
from nn.trainUtils import trainUnsupervised
from nn.profiler import setupLogging, Profiler

'''This is an example Stacked AutoEncoder used for unsupervised pre-training.
   The network topology should match that of the finalize Neural Network
   without having the output layer attached.
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
    parser.add_argument('--learnC', dest='learnC', type=float, default=.0031,
                        help='Rate of learning on Convolutional Layers.')
    parser.add_argument('--learnF', dest='learnF', type=float, default=.0015,
                        help='Rate of learning on Fully-Connected Layers.')
    parser.add_argument('--contrF', dest='contrF', type=float, default=.01,
                        help='Rate of contraction of the latent space on ' +
                             'Fully-Connected Layers.')
    parser.add_argument('--dropout', dest='dropout', type=bool, default=False,
                        help='Enable dropout throughout the network. Dropout '\
                             'percentages are based on optimal reported '\
                             'results. NOTE: Networks using dropout need to '\
                             'increase both neural breadth and learning rates')
    parser.add_argument('--kernel', dest='kernel', type=int, default=6,
                        help='Number of Convolutional Kernels in each Layer.')
    parser.add_argument('--neuron', dest='neuron', type=int, default=120,
                        help='Number of Neurons in Hidden Layer.')
    parser.add_argument('--epoch', dest='numEpochs', type=int, default=15,
                        help='Number of epochs to run per layer.')
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
    logName = 'cnnPreTrainer: ' + options.data
    log = setupLogging(logName, options.level, options.logfile)
    prof = Profiler(log=log, name=logName, profFile=options.profile)

    # create a random number generator for efficiency
    from numpy.random import RandomState
    from operator import mul
    rng = RandomState(int(time()))

    # NOTE: The pickleDataset will silently use previously created pickles if
    #       one exists (for efficiency). So watch out for stale pickles!
    train, test, labels = ingestImagery(filepath=options.data, shared=True,
                                        batchSize=options.batchSize, 
                                        holdoutPercentage=options.holdout, 
                                        log=log)
    trainShape = train[0].shape.eval()

    # create the stacked network -- LeNet-5 (minus the output layer)
    network = StackedAENetwork(train, prof=prof)

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

        # refactor the output to be (numImages*numKernels, 1, numRows, numCols)
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
                       reduce(mul, network.getNetworkOutputSize()[1:])),
            numNeurons=options.neuron, learningRate=options.learnF,
            dropout=.5 if options.dropout else 1., randomNumGen=rng))

        # the final output layer is removed from the normal NN --
        # the output layer is special, as it makes decisions about
        # patterns identified in previous layers, so it should only
        # be influenced/trained during supervised learning. 


    # train the SAE
    trainUnsupervised(network, __file__, options.data, 
                      numEpochs=options.numEpochs,
                      synapse=options.synapse, base=options.base, 
                      dropout=options.dropout, learnC=options.learnC,
                      learnF=options.learnF, contrF=options.contrF, 
                      kernel=options.kernel, neuron=options.neuron, log=log)

    # cleanup the network -- this ensures the profile is written
    del network