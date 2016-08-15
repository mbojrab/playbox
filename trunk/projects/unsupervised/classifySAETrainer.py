import argparse
from time import time
from six.moves import reduce

from ae.net import TrainerSAENetwork
from ae.contiguousAE import ContiguousAutoEncoder
from ae.convolutionalAE import ConvolutionalAutoEncoder
from dataset.ingest.labeled import ingestImagery
from nn.trainUtils import trainUnsupervised
from nn.profiler import setupLogging, Profiler
from dataset.minibatch import makeContiguous

def buildTrainerSAENetwork(train, target,
                           kernelConv, kernelSizeConv, downsampleConv, 
                           learnConv, momentumConv, dropoutConv,
                           neuronFull, learnFull, momentumFull, dropoutFull, 
                           prof=None) :
    '''Build the network in an automated way.'''
    import theano.tensor as t
    from operator import mul
    from numpy.random import RandomState
    rng = RandomState(int(time()))

    # create the stacked network -- LeNet-5 (minus the output layer)
    network = TrainerSAENetwork(train, target, prof=prof)

    if log is not None :
        log.info('Initialize the Network')

    # prepare for the next layer
    def prepare(network, count) :
        return (count + 1, 
                network.getNetworkOutputSize())

    layerCount = 1
    layerInputSize = train[0].shape.eval()[1:]
    for k,ks,do,l,m,dr in zip(kernelConv, kernelSizeConv, downsampleConv, 
                              learnConv, momentumConv, dropoutConv) :
        # add a convolutional layer as defined
        network.addLayer(ConvolutionalAutoEncoder(
            layerID='conv' + str(layerCount), 
            inputSize=layerInputSize, kernelSize=(k,layerInputSize[1],ks,ks),
            downsampleFactor=[do,do], dropout=dr, learningRate=l,
            activation=t.nnet.sigmoid, randomNumGen=rng))

        # prepare for the next layer
        layerCount, layerInputSize = prepare(network, layerCount)

    # update to transition for fully connected layers
    layerInputSize = (layerInputSize[0], reduce(mul, layerInputSize[1:]))
    for n,l,m,dr in zip(neuronFull, learnFull, momentumFull, dropoutFull) :
        # add a fully-connected layer as defined
        network.addLayer(ContiguousAutoEncoder(
            layerID='fully' + str(layerCount), 
            inputSize=layerInputSize, numNeurons=n, learningRate=l,
            activation=t.nnet.sigmoid, dropout=dr, randomNumGen=rng))            

        # prepare for the next layer
        layerCount, layerInputSize = prepare(network, layerCount)

    return network

def readTargetData(targetpath) :
    '''Read a directory of data to use as a feature matrix.'''
    import os
    from dataset.reader import readImage
    return makeContiguous([(readImage(os.path.join(targetpath, im))) \
                           for im in os.listdir(targetpath)])[0]

def testCloseness(net, imagery) :
    '''Test the imagery for how close it is to the target data. This also sorts
       the results according to closeness, so we can create a tiled tip-sheet.
    '''
    for x in imagery :
        print(net.closeness(x))
    #closenessImagery = makeContiguous([(x, net.closeness(x)) for x in imagery])
    #print(closenessImagery)
    # TODO: Rank the results in order of closeness
    #return closenessImagery

if __name__ == '__main__' :
    '''Build and train an SAE, then test a '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--log', dest='logfile', type=str, default=None,
                        help='Specify log output file.')
    parser.add_argument('--level', dest='level', default='INFO', type=str, 
                        help='Log Level.')
    parser.add_argument('--prof', dest='profile', type=str, 
                        default='Application-Profiler.xml',
                        help='Specify profile output file.')
    parser.add_argument('--kernel', dest='kernel', type=list,
                        default=[100, 200],
                        help='Number of Convolutional Kernels in each Layer.')
    parser.add_argument('--kernelSize', dest='kernelSize', type=list,
                        default=[5, 5],
                        help='Size of Convolutional Kernels in each Layer.')
    parser.add_argument('--downsample', dest='downsample', type=list,
                        default=[2, 2],
                        help='Downsample factor in each Convolutional Layer.')
    parser.add_argument('--learnC', dest='learnC', type=list,
                        default=[.08, .08],
                        help='Rate of learning on Convolutional Layers.')
    parser.add_argument('--momentumC', dest='momentumC', type=list,
                        default=[.5, .5],
                        help='Rate of momentum on Convolutional Layers.')
    parser.add_argument('--dropoutC', dest='dropoutC', type=list, 
                        default=[0.8, 0.5],
                        help='Dropout amount for the Convolutional Layer.')
    parser.add_argument('--neuron', dest='neuron', type=list, 
                        default=[500, 300, 100],
                        help='Number of Neurons in Hidden Layer.')
    parser.add_argument('--learnF', dest='learnF', type=list,
                        default=[.02, .02, .02],
                        help='Rate of learning on Fully-Connected Layers.')
    parser.add_argument('--momentumF', dest='momentumF', type=list,
                        default=[.2, .2, .2],
                        help='Rate of momentum on Fully-Connected Layers.')
    parser.add_argument('--dropoutF', dest='dropoutF', type=list, 
                        default=[0.5, 0.5, 1],
                        help='Dropout amount for the Fully-Connected Layer.')
    parser.add_argument('--epoch', dest='numEpochs', type=int, default=15,
                        help='Number of epochs to run per layer.')
    parser.add_argument('--batch', dest='batchSize', type=int, default=100,
                        help='Batch size for training and test sets.')
    parser.add_argument('--base', dest='base', type=str, default='./saeClass',
                        help='Base name of the network output and temp files.')
    parser.add_argument('--target', dest='targetDir', type=str, required=True,
                        help='Directory with target data to match.')
    parser.add_argument('--syn', dest='synapse', type=str, default=None,
                        help='Load from a previously saved network.')
    parser.add_argument('data', help='Directory or pkl.gz file for the ' +
                                     'training and test sets')
    options = parser.parse_args()

    # setup the logger
    logName = 'SAE-Classification Benchmark:  ' + options.data
    log = setupLogging(logName, options.level, options.logfile)
    prof = Profiler(log=log, name=logName, profFile=options.profile)

    # NOTE: The pickleDataset will silently use previously created pickles if
    #       one exists (for efficiency). So watch out for stale pickles!
    train, test, labels = ingestImagery(filepath=options.data, shared=True,
                                        batchSize=options.batchSize, log=log)

    # load example imagery --
    # these are confirmed objects we are attempting to identify 
    target = readTargetData(options.targetDir)

    if options.synapse is not None :
        # load a previously saved network
        network = TrainerSAENetwork(train, target, prof=prof)
        network.load(options.synapse)

    else :
        network = buildTrainerSAENetwork(train, target, prof=prof,
                                         kernelConv=options.kernel, 
                                         kernelSizeConv=options.kernelSize, 
                                         downsampleConv=options.downsample, 
                                         learnConv=options.learnC, 
                                         momentumConv=options.momentumC,
                                         dropoutConv=options.dropoutC,
                                         neuronFull=options.neuron, 
                                         learnFull=options.learnF, 
                                         momentumFull=options.momentumF,
                                         dropoutFull=options.dropoutF)

    # train the SAE
    trainUnsupervised(network, __file__, options.data, 
                      numEpochs=options.numEpochs, synapse=options.synapse,
                      base=options.base, dropout=(len(options.dropoutC)>0),
                      learnC=options.learnC, learnF=options.learnF,
                      contrF=None, kernel=options.kernel,
                      neuron=options.neuron, log=log)

    # test the training data for similarity to the target
    results = testCloseness(network, test[0].get_value(borrow=True))

    # TODO: create the ordered tip-sheet
