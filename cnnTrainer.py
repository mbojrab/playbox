import theano, numpy
import theano.tensor as t
from net import Net
from contiguousLayer import ContiguousLayer
from convolutionalLayer import ConvolutionalLayer

def ingestImagery(filepath=None, log=None) :
    '''Load the dataset provided by the user.
       filepath : This can be a cPickle, a path to the directory structure,
                  or None if the MNIST dataset should be loaded.
       log      : Logger for tracking the progress
       return   :
           Format -- 
           [(trainData, trainLabel), (testData, testLabel)], labels

           The 'trainLabel' and 'testLabel' are integer values corresponding
           to the index into the 'labels' string vector. this provides a
           better means to identify errors during back propagation, but
           still allows label finding during classification.
    '''
    import os, cPickle, gzip

    # if None load the MNIST
    filepath = 'mnist.pkl.gz' if None else filepath
    train = test = None

    log.info('Ingesting imagery...')

    # the mnist dataset is a special case
    if 'mnist.pkl.gz' in filepath :

        # see if we have previously downloaded the file
        if filepath not in os.listdir(os.getcwd()) :
            import urllib
            url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
            log.debug('Downloading data from ' + url)
            urllib.urlretrieve(url, filepath)

        # Load the dataset to memory -- 
        # the mnist dataset is a special case created by University of Toronto
        log.debug('Load the data into memory')
        with gzip.open(filepath, 'rb') as f :
            # this dataset has a valid and test and no labels. 
            train, valid, test = cPickle.load(f)

            # add the validation set to the training set
            log.debug('Combine the Train and Valid datasets')
            trainData, trainLabel = train
            validData, validLabel = valid
            train = trainData + validData, trainLabel + validLabel

            # create a label vector
            log.debug('Create the labels')
            labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    else :
        # Load the dataset to memory
        log.debug('Load the data into memory')
        with gzip.open(filepath, 'rb') as f :
            train, test, labels = cPickle.load(f)

    def loadShared(x, borrow=True) :
        return theano.shared(numpy.asarray(
            x, dtype=theano.config.floatX), borrow=borrow)
    def splitToShared(x, borrow=True) :
        data, label = x
        return (loadShared(data), t.cast(loadShared(label), 'int32'))

    # load each into shared variables -- 
    # this avoids having to copy the data to the GPU between each call
    log.debug('Transfer the memory into shared variables')
    return [splitToShared(train), splitToShared(test)], labels

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
        network.getNetworkOutput()[1], len(test.labels),
        float(options.learnF), runCPU=runCPU, randomNumGen=rng))


    [train, test], labels = ingestImagery(options.data, log) :


    lastBest = 0
    globalCount = 0
    degradationCount = 0
    runningAccuracy = 0.0
    while True :
        # run the specified number of epochs
        for ii in range(int(options.limit)) :
            network.train(train.inputs[ii], train.labels[ii])

        # calculate the accuracy against the test set
        runAcc = 0.0
        for input, label in zip(train.inputs, train.labels) :
            if network.classify(input).argmax() == label :
                runAcc += 1.0
        runAcc /= float(len(train.labels))

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
        globalCount += 1


    # test the classify runtime
    print "Classifying Inputs..."
    timer = time()
    for i in range(numRuns) :
        out = network.classify(arr)
    timer = time() - timer
    print "total time: " + str(timer) + \
          "s | per input: " + str(timer/numRuns) + "s"
    print (out.argmax(), out)

    # test the train runtime
    numRuns = 100000
    print "Training Network..."
    timer = time()
    for i in range(numRuns) :
        network.train(arr, exp)
        print network.classify(arr)
    timer = time() - timer
    print "total time: " + str(timer) + \
          "s | per input: " + str(timer/numRuns) + "s"
    out = network.classify(arr)
    print (out.argmax(), out)
