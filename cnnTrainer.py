import theano.tensor as t
from net import Net
from contiguousLayer import ContiguousLayer
from convolutionalLayer import ConvolutionalLayer
import datasetUtils 

'''
'''
if __name__ == '__main__' :
    import argparse, logging
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', dest='logfile',
                        help='Specify log output file.')
    parser.add_argument('--level', dest='level',
                        help='Log Level.')
    parser.add_argument('--learnC', dest='learnC',
                        help='Rate of learning on Convolutional Layers.')
    parser.add_argument('--learnF', dest='learnF',
                        help='Rate of learning on Fully-Connected Layers.')
    parser.add_argument('--kernel', dest='kernel',
                        help='Number of Convolutional Kernels in each Layer.')
    parser.add_argument('--neuron', dest='neuron',
                        help='Number of Neurons in Hidden Layer.')
    parser.add_argument('--limit', dest='limit',
                        help='Number of runs between validation checks')
    parser.add_argument('--stop', dest='stop',
                        help='Number of inferior validation checks before ending')
    parser.add_argument('--cpu', dest='runCPU', help='Run training on the CPU')
    parser.add_argument('data', dest='data',
                        help='Pickle file for the training and test sets')
    options = parser.parse_args()

    # this makes the indexing more intuitive
    DATA  = 0
    LABEL = 1

    # setup the logger
    log = logging.getLogger('cnnTrainer: ' + options.train)
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
    rng = RandomState(int(time()))

    input = t.fvector('input')
    expectedOutput = t.bvector('expectedOutput')

    train, test, labels = datasetUtils.ingestImagery(
        datasetUtils.pickleDataset(options.data, log=log), log=log)

    # create the network -- LeNet-5
    runCPU = options.runCPU.upper() == 'TRUE'
    network = Net(regType='', runCPU=False, log=log)

    # add convolutional layers
    network.addLayer(ConvolutionalLayer('c1', input, (1,1,28,28), (6,5,5),
                                        (2,2), runCPU=runCPU, randomNumGen=rng,
                                        learningRate=options.learnC))
    network.addLayer(ConvolutionalLayer('c2', network.getNetworkOutput(),
                                        network.getOutputSize(), (6,5,5),
                                        (2,2), runCPU=runCPU, randomNumGen=rng,
                                        learningRate=options.learnC))
    # add fully connected layers
    network.addLayer(ContiguousLayer(
        'f3', network.getNetworkOutput()[0].flatten(2),
        network.getNetworkOutput()[1], int(options.neuron),
        float(options.learnF), runCPU=runCPU, randomNumGen=rng))
    network.addLayer(ContiguousLayer(
        'f4', network.getNetworkOutput()[0],
        network.getNetworkOutput()[1], len(labels),
        float(options.learnF), runCPU=runCPU, randomNumGen=rng))

    lastBest = 0
    globalCount = 0
    degradationCount = 0
    runningAccuracy = 0.0
    while True :
        # run the specified number of epochs
        numEpochs = int(options.limit)
        for localEpoch in range(numEpochs) :
            for ii in range(len(train[DATA])) :
                network.train(train[DATA][ii], train[LABEL][ii])

        # calculate the accuracy against the test set
        runAcc = 0.0
        for input, label in zip(test[DATA], test[LABEL]) :
            if network.classify(input).argmax() == label :
                runAcc += 1.0
        runAcc /= float(len(test[LABEL]))

        # check if we've done better
        if runAcc > runningAccuracy :
            # reset and save the network
            degradationCount = 0
            lastBest = globalCount
            network.save('')
        else :
            # quit once we've had 'stop' times of less accuracy
            degradationCount += 1
            if degradationCount > options.stop :
                break
        globalCount += numEpochs

