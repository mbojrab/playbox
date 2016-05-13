import theano.tensor as t
from nn.net import TrainerNetwork as Net
from nn.contiguousLayer import ContiguousLayer
from nn.convolutionalLayer import ConvolutionalLayer
from nn.datasetUtils import ingestImagery, pickleDataset, splitToShared
import os, argparse, logging
from time import time

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

    # this makes the indexing more intuitive
    DATA, LABEL = 0, 1

    # setup the logger
    log = logging.getLogger('cnnTrainer: ' + options.data)
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
    trainSize = train[0].shape

    tr = splitToShared(train, borrow=True)
    te = splitToShared(test,  borrow=True)

    # create the network -- LeNet-5
    network = Net(train, te, labels, regType='L2',
                  regScaleFactor=1. / (options.kernel + options.kernel + 
                                       options.neuron + len(labels)), 
                  log=log)

    if options.synapse is not None :
        # load a previously saved network
        network.load(options.synapse)
    else :
        log.info('Initializing Network...')
        input = t.ftensor4('input')

        # add convolutional layers
        network.addLayer(ConvolutionalLayer(
            layerID='c1', input=input, 
            inputSize=trainSize[1:],
            kernelSize=(options.kernel,trainSize[2],5,5),
            downsampleFactor=(2,2), randomNumGen=rng,
            dropout=.8 if options.dropout else 1.,
            learningRate=options.learnC, momentumRate=options.momentum))

        # refactor the output to be (numImages*numKernels, 1, numRows, numCols)
        # this way we don't combine the channels kernels we created in 
        # the first layer and destroy our dimensionality
        network.addLayer(ConvolutionalLayer(
            layerID='c2', input=network.getNetworkOutput(), 
            inputSize=network.getNetworkOutputSize(), 
            kernelSize=(options.kernel,options.kernel,5,5),
            downsampleFactor=(2,2), randomNumGen=rng,
            dropout=.5 if options.dropout else 1.,
            learningRate=options.learnC, momentumRate=options.momentum))

        # add fully connected layers
        network.addLayer(ContiguousLayer(
            layerID='f3', input=network.getNetworkOutput(),
            inputSize=(network.getNetworkOutputSize()[0], 
                       reduce(mul, network.getNetworkOutputSize()[1:])),
            numNeurons=options.neuron, 
            learningRate=options.learnF, momentumRate=options.momentum,
            dropout=.5 if options.dropout else 1., randomNumGen=rng))
        network.addLayer(ContiguousLayer(
            layerID='f4', input=network.getNetworkOutput(),
            inputSize=network.getNetworkOutputSize(), numNeurons=len(labels),
            learningRate=options.learnF, momentumRate=options.momentum,
            activation=None, randomNumGen=rng))

    globalCount = lastBest = degradationCount = 0
    numEpochs = options.limit
    runningAccuracy = 0.0
    lastSave = ''
    while True :
        timer = time()

        # run the specified number of epochs
        globalCount = network.trainEpoch(globalCount, numEpochs)
        # calculate the accuracy against the test set
        curAcc = network.checkAccuracy()
        log.info('Checking Accuracy - {0}s ' \
                 '\n\tCorrect   : {1}% \n\tIncorrect : {2}%'.format(
                 time() - timer, curAcc, (100-curAcc)))

        # check if we've done better
        if curAcc > runningAccuracy :
            # reset and save the network
            degradationCount = 0
            runningAccuracy = curAcc
            lastBest = globalCount
            lastSave = options.base + \
                       '_dropout'+ str(options.dropout) + \
                       '_learnC' + str(options.learnC) + \
                       '_learnF' + str(options.learnF) + \
                       '_momentum' + str(options.momentum) + \
                       '_kernel' + str(options.kernel) + \
                       '_neuron' + str(options.neuron) + \
                       '_epoch' + str(lastBest) + '.pkl.gz'
            network.save(lastSave)
        else :
            # increment the number of poor performing runs
            degradationCount += 1

        # stopping conditions for regularization
        if degradationCount > int(options.stop) or runningAccuracy == 100. :
            break

    # rename the network which achieved the highest accuracy
    bestNetwork = options.base + '_FinalOnHoldOut_' + \
                  os.path.basename(options.data) + '_epoch' + str(lastBest) + \
                  '_acc' + str(runningAccuracy) + '.pkl.gz'
    log.info('Renaming Best Network to [' + bestNetwork + ']')
    if os.path.exists(bestNetwork) :
        os.remove(bestNetwork)
    os.rename(lastSave, bestNetwork)
