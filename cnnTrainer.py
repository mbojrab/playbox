import theano.tensor as t
from net import Net
from contiguousLayer import ContiguousLayer
from convolutionalLayer import ConvolutionalLayer
import datasetUtils, numpy, os

'''
'''
if __name__ == '__main__' :
    import argparse, logging
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', dest='logfile', default=None,
                        help='Specify log output file.')
    parser.add_argument('--level', dest='level', default='INFO',
                        help='Log Level.')
    parser.add_argument('--learnC', dest='learnC', default=.0031,
                        help='Rate of learning on Convolutional Layers.')
    parser.add_argument('--learnF', dest='learnF', default=.0015,
                        help='Rate of learning on Fully-Connected Layers.')
    parser.add_argument('--momentum', dest='momentum', default=.3,
                        help='Momentum rate all layers.')
    parser.add_argument('--kernel', dest='kernel', default=6,
                        help='Number of Convolutional Kernels in each Layer.')
    parser.add_argument('--neuron', dest='neuron', default=120,
                        help='Number of Neurons in Hidden Layer.')
    parser.add_argument('--limit', dest='limit', default=5,
                        help='Number of runs between validation checks')
    parser.add_argument('--stop', dest='stop', default=5,
                        help='Number of inferior validation checks before ending')
    parser.add_argument('--base', dest='base', default='./leNet5',
                        help='Base name of the network output and temp files.')
    parser.add_argument('--syn', dest='synapse', default=None,
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
    from time import time
    from operator import mul
    rng = RandomState(int(time()))

    # NOTE: The pickleDataset will silently use previously created pickles if
    #       one exists (for efficiency). So watch out for stale pickles!
    train, test, labels = datasetUtils.ingestImagery(
        datasetUtils.pickleDataset(options.data, log=log), log=log)

    # create the network -- LeNet-5
    network = Net(regType='', log=log)

    if options.synapse is not None :
        # load a previously saved network
        network.load(options.synapse)
    else :
        log.info('Initializing Network...')

        input = t.ftensor4('input')
        expectedOutput = t.bvector('expectedOutput')

        # add convolutional layers
        network.addLayer(ConvolutionalLayer(
            layerID='c1', input=input, inputSize=(1,1,28,28), kernelSize=(6,1,5,5),
            downsampleFactor=(2,2), randomNumGen=rng, learningRate=options.learnC))
        # refactor the output to be (numImages*numKernels, 1, numRows, numCols)
        # this way we don't combine the channels kernels we created in the first
        # layer and destroy our dimensionality
        netOutputSize = network.getNetworkOutputSize()
        netOutputSize = (netOutputSize[0] * netOutputSize[1], 1,
                         netOutputSize[2], netOutputSize[3])
        network.addLayer(ConvolutionalLayer(
            layerID='c2', input=network.getNetworkOutput().reshape(netOutputSize),
            inputSize=netOutputSize, kernelSize=(6,1,5,5), downsampleFactor=(2,2), 
            randomNumGen=rng, learningRate=options.learnC))

        # add fully connected layers
        network.addLayer(ContiguousLayer(
            layerID='f3', input=network.getNetworkOutput().flatten(),
            inputSize=reduce(mul, network.getNetworkOutputSize()),
            numNeurons=int(options.neuron), learningRate=float(options.learnF),
            randomNumGen=rng))
        network.addLayer(ContiguousLayer(
            layerID='f4', input=network.getNetworkOutput(),
            inputSize=network.getNetworkOutputSize(), numNeurons=len(labels),
            learningRate=float(options.learnF), randomNumGen=rng))

    globalCount = lastBest = degradationCount = 0
    runningAccuracy = 0.0
    lastSave = ''
    expectedOutput = numpy.zeros(network.getNetworkOutputSize(), dtype='int32')
    while True :
        from time import time

        # run the specified number of epochs
        numEpochs = int(options.limit)
        for localEpoch in range(numEpochs) :
            timer = time()
            for ii in range(len(train[LABEL])) :
                expectedOutput[train[LABEL][ii]] = 1
                network.train(train[DATA][ii], expectedOutput)
                expectedOutput[train[LABEL][ii]] = 0
            log.info('Training Epoch [' + str(globalCount + localEpoch) + 
                     '] - ' + str(time() - timer) + 's')

        # calculate the accuracy against the test set
        curAcc = 0.0
        timer = time()
        for input, label in zip(test[DATA], test[LABEL]) :
            if network.classify(input).argmax() == label :
                curAcc += 1.0
        curAcc /= float(len(test[LABEL]))
        log.info('Testing Accuracy - ' + str(time() - timer) + 's\n' +
                 '\tCorrect: ' + str(curAcc * 100) + '%\n' +
                 '\tFalse  : ' + str((1-curAcc) * 100) + '%\n')

        # check if we've done better
        if curAcc > runningAccuracy :
            # reset and save the network
            degradationCount = 0
            runningAccuracy = curAcc
            lastBest = globalCount
            lastSave = options.base + \
                       '_learnC' + str(options.learnC) + \
                       '_learnF' + str(options.learnF) + \
                       '_momentum' + str(options.momentum) + \
                       '_kernel' + str(options.kernel) + \
                       '_neuron' + str(options.neuron) + \
                       '_epoch' + str(lastBest) + '.pkl.gz'
            network.save(lastSave)
        else :
            degradationCount += 1

        globalCount += numEpochs

        # stopping conditions for regularization
        if degradationCount > int(options.stop) or runningAccuracy == 1. :
            break

    # rename the network which achieved the highest accuracy
    bestNetwork = options.base + '_FinalOnHoldOut_' + \
                  os.path.basename(options.data) + '_epoch' + str(lastBest) + \
                  '_acc' + str(runningAccuracy) + '.pkl.gz'
    log.info('Renaming Best Network to [' + bestNetwork + ']')
    os.rename(lastSave, bestNetwork)
