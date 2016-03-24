import theano.tensor as t
from ae.net import StackedAENetwork
from ae.contiguousAE import ContractiveAutoEncoder
from ae.convolutionalAE import ConvolutionalAutoEncoder
from nn.datasetUtils import ingestImagery, pickleDataset, splitToShared
import os, argparse, logging
from time import time

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
    parser.add_argument('--learnC', dest='learnC', type=float, default=.0031,
                        help='Rate of learning on Convolutional Layers.')
    parser.add_argument('--learnF', dest='learnF', type=float, default=.0015,
                        help='Rate of learning on Fully-Connected Layers.')
    parser.add_argument('--contrF', dest='contrF', type=float, default=.01,
                        help='Rate of contraction of the latent space on ' +
                             'Fully-Connected Layers.')
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

    # this makes the indexing more intuitive
    DATA, LABEL = 0, 1

    # setup the logger
    log = logging.getLogger('cnnPreTrainer: ' + options.data)
    log.setLevel(options.level.upper())
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    stream = logging.StreamHandler()
    stream.setLevel(options.level.upper())
    stream.setFormatter(formatter)
    log.addHandler(stream)
    if options.logfile is not None :
        logFile = logging.FileHandler(options.logfile)
        logFile.setLevel(options.level.upper())
        logFile.setFormatter(formatter)
        log.addHandler(logFile)

    # create a random number generator for efficiency
    from numpy.random import RandomState
    from operator import mul
    rng = RandomState(int(time()))

    # NOTE: The pickleDataset will silently use previously created pickles if
    #       one exists (for efficiency). So watch out for stale pickles!
    train, test, labels = ingestImagery(pickleDataset(
            options.data, batchSize=options.batchSize, 
            holdoutPercentage=options.holdout, log=log),
        shared=False, log=log)
    trainShape = train[0].shape

    # create the stacked network -- LeNet-5 (minus the output layer)
    network = StackedAENetwork(splitToShared(train, borrow=True), log=log)

    if options.synapse is not None :
        # load a previously saved network
        network.load(options.synapse)
    else :
        log.info('Initializing Network...')
        input = t.ftensor4('input')

        # add convolutional layers
        network.addLayer(ConvolutionalAutoEncoder(
            layerID='c1', input=input, 
            inputSize=trainShape[1:], 
            kernelSize=(options.kernel,trainShape[2],5,5),
            downsampleFactor=(2,2), randomNumGen=rng,
            learningRate=options.learnC))

        # refactor the output to be (numImages*numKernels, 1, numRows, numCols)
        # this way we don't combine the channels kernels we created in 
        # the first layer and destroy our dimensionality
        network.addLayer(ConvolutionalAutoEncoder(
            layerID='c2',
            input=network.getNetworkOutput(), 
            inputSize=network.getNetworkOutputSize(), 
            kernelSize=(options.kernel,options.kernel,5,5),
            downsampleFactor=(2,2), randomNumGen=rng,
            learningRate=options.learnC))

        # add fully connected layers
        network.addLayer(ContractiveAutoEncoder(
            layerID='f3', input=network.getNetworkOutput().flatten(2),
            inputSize=(network.getNetworkOutputSize()[0], 
                       reduce(mul, network.getNetworkOutputSize()[1:])),
            numNeurons=options.neuron, learningRate=options.learnF,
            randomNumGen=rng))

        # the final output layer is removed from the normal NN --
        # the output layer is special, as it makes decisions about
        # patterns identified in previous layers, so it should only
        # be influenced/trained during supervised learning. 
    network.writeWeights(-1)

    # train each layer in sequence --
    # first we pre-train the data and at each epoch, we save it to disk
    lastSave = ''
    for layerIndex in range(network.getNumLayers()) :
        globalEpoch = 0
        globalEpoch, cost = network.trainEpoch(layerIndex, globalEpoch, 
                                               options.numEpochs)
        network.writeWeights(globalEpoch)
        lastSave = options.base + \
                   '_learnC' + str(options.learnC) + \
                   '_learnF' + str(options.learnF) + \
                   '_contrF' + str(options.contrF) + \
                   '_kernel' + str(options.kernel) + \
                   '_neuron' + str(options.neuron) + \
                   '_layer' + str(layerIndex) + \
                   '_epoch' + str(globalEpoch) + '.pkl.gz'
        network.save(lastSave)

    # rename the network which achieved the highest accuracy
    bestNetwork = options.base + '_PreTrained_' + \
                  os.path.basename(options.data) + '_epoch' + \
                  str(options.numEpochs) + '.pkl.gz'
    log.info('Renaming Best Network to [' + bestNetwork + ']')
    if os.path.exists(bestNetwork) :
        os.remove(bestNetwork)
    os.rename(lastSave, bestNetwork)

    # cleanup the network -- this ensures the profile is written
    del network