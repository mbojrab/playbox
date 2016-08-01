import argparse, os

from nn.net import TrainerNetwork
from nn.contiguousLayer import ContiguousLayer
from nn.convolutionalLayer import ConvolutionalLayer
from dataset.ingest.labeled import ingestImagery
from nn.trainUtils import trainSupervised

from distill.net import DistilleryTrainer
from nn.profiler import setupLogging, Profiler

def createNetwork(inputSize, numKernels, numNeurons, numLabels) :
    from nn.net import ClassifierNetwork
    from six.moves import reduce

    localPath = './local.pkl.gz'
    network = ClassifierNetwork()

    lr = [.08, .05, .02]
    mr = [.8, .8, .8]

    # add convolutional layers
    network.addLayer(ConvolutionalLayer(
        layerID='c1', inputSize=inputSize,
        kernelSize=(numKernels,inputSize[1],3,3),
        downsampleFactor=(3,3), randomNumGen=rng,
        learningRate=lr[0], momentumRate=mr[0]))
    # add fully connected layers
    network.addLayer(ContiguousLayer(
        layerID='f2', 
        inputSize=(network.getNetworkOutputSize()[0],
                   reduce(mul, network.getNetworkOutputSize()[1:])),
        numNeurons=numNeurons, randomNumGen=rng,
        learningRate=lr[1], momentumRate=mr[1]))
    network.addLayer(ContiguousLayer(
        layerID='f3', inputSize=network.getNetworkOutputSize(), 
        numNeurons=numLabels, learningRate=lr[2], momentumRate=mr[2], 
        activation=None, randomNumGen=rng))

    # save it to disk in order to load it into both networks
    network.save(localPath)

    return localPath


'''This application will test the performance of a distilled network vs the 
   baseline with only the one-hot labels. Performance will be measured both
   in inference-time speed gains and in accuracy from the original network.
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
    parser.add_argument('--kernel', dest='kernel', type=int, default=6,
                        help='Number of Convolutional Kernels in each Layer.')
    parser.add_argument('--neuron', dest='neuron', type=int, default=120,
                        help='Number of Neurons in Hidden Layer.')
    parser.add_argument('--limit', dest='limit', type=int, default=5,
                        help='Number of runs between validation checks.')
    parser.add_argument('--stop', dest='stop', type=int, default=5,
                        help='Number of inferior validation checks to end.')
    parser.add_argument('--softness', dest='softness', type=float, default=4.0,
                        help='Softness factor in softmax function.')
    parser.add_argument('--factor', dest='factor', type=float, default=0.8,
                        help='Factor of error coming from deep transfer.')
    parser.add_argument('--holdout', dest='holdout', type=float, default=.05,
                        help='Percent of data to be held out for testing.')
    parser.add_argument('--batch', dest='batchSize', type=int, default=50,
                        help='Batch size for training and test sets.')
    parser.add_argument('--base', dest='base', type=str, default='./distillery',
                        help='Base name of the network output and temp files.')
    parser.add_argument('--deep', dest='deep', type=str, default=None,
                        help='Synapse for the deep network to distill. This ' +
                        'network should be trained and ready.')
    parser.add_argument('--syn', dest='synapse', type=str, default=None,
                        help='Load from a previously saved network.')
    parser.add_argument('data', help='Directory or pkl.gz file for the ' +
                                     'training and test sets. This can be ' + 
                                     'the location of a dark pickle.')
    options = parser.parse_args()

    # setup the logger
    logName = 'distillery: ' + options.data
    log = setupLogging(logName, options.level, options.logfile)
    prof = Profiler(log=log, name=logName, profFile=options.profile)

    # create a random number generator for efficiency
    from numpy.random import RandomState
    from operator import mul
    from time import time
    rng = RandomState(int(time()))
    #rng = RandomState(4567) # always initialize the same

    # NOTE: The pickleDataset will silently use previously created pickles if
    #       one exists (for efficiency). So watch out for stale pickles!
    # NOTE: User may pass a dark pickle into here, and the logic will react
    #       appropriately to the situation.
    train, test, labels = ingestImagery(filepath=options.data, shared=True,
                                        batchSize=options.batchSize,
                                        holdoutPercentage=options.holdout,
                                        log=log)
    inputSize = train[0].shape.eval()

    # create a file with pre-initialized weights so both networks use the same
    # baseline for testing.
    if options.synapse is None :
        networkFile = createNetwork(inputSize=inputSize[1:],
                                    numKernels=options.kernel,
                                    numNeurons=options.neuron,
                                    numLabels=len(labels))
    else :
        networkFile = options.synapse

    # load the nn.net.TrainerNetwork
    baseNet = TrainerNetwork(
                  train[:2], test, labels, regType='L2',
                  regScaleFactor=1. / (options.kernel + options.kernel + 
                                       options.neuron + len(labels)),
                  filepath=networkFile, prof=prof)

    # perform baseline training
    baseFile = trainSupervised(baseNet, __file__, options.data, 
                               numEpochs=options.limit, stop=options.stop, 
                               base=options.base + '_baseline', log=log)

    # load the distill.DistilleryTrainer
    distNet = DistilleryTrainer(
                  train, test, labels, regType='L2',
                  regScaleFactor=1. / (options.kernel + options.kernel + 
                                       options.neuron + len(labels)),
                  softmaxTemp=options.softness, transFactor=options.factor,
                  filepath=networkFile, prof=prof)
    # user has not specified a dark pickle infused with additional knowledge
    # from a deep network. In this case, we had the deep network directly to
    # the object in order to get the soft targets JIT
    if len(train) == 2 :
        distNet.loadDeepNetwork(options.deep)

    # perform distilled training
    distFile = trainSupervised(distNet, __file__, options.data, 
                               numEpochs=options.limit, stop=options.stop, 
                               base=options.base + '_distilled', log=log)

    # load a new network to collect statistics
    network = TrainerNetwork(
                train[:2], test, labels, regType='L2',
                regScaleFactor=1. / (options.kernel + options.kernel + 
                                     options.neuron + len(labels)), 
                prof=prof)

    # collect the statistics on performance
    prof.startProfile('Checking Statistics')
    accStat = -1.0
    for f in (baseFile, distFile) :
        prof.startProfile('Loading [' + f + ']')
        network.load(f)
        timer = time()
        curAcc = network.checkAccuracy()
        log.info('Checking Accuracy - {0}s ' \
                 '\n\tCorrect   : {1}% \n\tIncorrect : {2}%'.format(
                 time() - timer, curAcc, (100-curAcc)))
        prof.endProfile()

    # cleanup the area
    if options.synapse is None :
        os.remove(networkFile)
