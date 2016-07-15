import theano.tensor as t
import argparse
from time import time
from six.moves import reduce

from ae.net import StackedAENetwork
from ae.contiguousAE import ContiguousAutoEncoder
from ae.convolutionalAE import ConvolutionalAutoEncoder
from dataset.ingest.unlabeled import ingestImagery
from dataset.chip import randomChip
from nn.trainUtils import trainUnsupervised
from nn.profiler import setupLogging

def buildStackedAENetwork(train,
                          kernelConv, kernelSizeConv, downsampleConv, 
                          learnConv, momentumConv, dropoutConv,
                          neuronFull, learnFull, momentumFull, dropoutFull, 
                          log=None) :
    from operator import mul
    from numpy.random import RandomState
    rng = RandomState(int(time()))

    # create the stacked network -- LeNet-5 (minus the output layer)
    network = StackedAENetwork(train, log=log)

    if log is not None :
        log.info('Initialize the Network')

    # prepare for the next layer
    def prepare(network, count) :
        return (count + 1, 
                network.getNetworkOutputSize())

    layerCount = 1
    layerInputSize = train.eval().shape[1:]
    for k,ks,do,l,m,dr in zip(kernelConv, kernelSizeConv, downsampleConv, 
                              learnConv, momentumConv, dropoutConv) :
        # add a convolutional layer as defined
        network.addLayer(ConvolutionalAutoEncoder(
            layerID='conv' + str(layerCount), 
            inputSize=layerInputSize, kernelSize=(k,layerInputSize[1],ks,ks),
            downsampleFactor=[do,do], dropout=dr, learningRate=l,
            randomNumGen=rng))

        # prepare for the next layer
        layerCount, layerInputSize = prepare(network, layerCount)

    # update to transition for fully connected layers
    layerInputSize = (layerInputSize[0], reduce(mul, layerInputSize[1:]))
    for n,l,m,dr in zip(neuronFull, learnFull, momentumFull, dropoutFull) :
        # add a fully-connected layer as defined
        network.addLayer(ContiguousAutoEncoder(
            layerID='fully' + str(layerCount), 
            inputSize=layerInputSize, numNeurons=n, learningRate=l,
            dropout=dr, randomNumGen=rng))            

        # prepare for the next layer
        layerCount, layerInput, layerInputSize = prepare(network, layerCount)

    return network


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
    parser.add_argument('--kernel', dest='kernel', type=list,
                        default=[500, 1000, 1000],
                        help='Number of Convolutional Kernels in each Layer.')
    parser.add_argument('--kernelSize', dest='kernelSize', type=list,
                        default=[3, 5, 5],
                        help='Size of Convolutional Kernels in each Layer.')
    parser.add_argument('--downsample', dest='downsample', type=list,
                        default=[1, 2, 2],
                        help='Downsample factor in each Convolutional Layer.')
    parser.add_argument('--learnC', dest='learnC', type=list,
                        default=[.0031, .0031, .0031],
                        help='Rate of learning on Convolutional Layers.')
    parser.add_argument('--dropoutC', dest='dropoutC', type=list, 
                        default=[0.8, 0.5, 0.5],
                        help='Dropout amount for the Convolutional Layer.')
    parser.add_argument('--neuron', dest='neuron', type=list, 
                        default=[1500, 500, 100],
                        help='Number of Neurons in Hidden Layer.')
    parser.add_argument('--learnF', dest='learnF', type=list,
                        default=[.0015, .0015, .0015],
                        help='Rate of learning on Fully-Connected Layers.')
    parser.add_argument('--dropoutF', dest='dropoutF', type=list, 
                        default=[0.5, 0.5, 1],
                        help='Dropout amount for the Fully-Connected Layer.')
    parser.add_argument('--chipSize', dest='chipSize', type=list, nargs=2,
                        default=[50,50], 
                        help='Number of epochs to run per layer.')
    parser.add_argument('--numChips', dest='numChips', type=int, default=1000,
                        help='Number of epochs to run per layer.')
    parser.add_argument('--epoch', dest='numEpochs', type=int, default=15,
                        help='Number of epochs to run per layer.')
    parser.add_argument('--batch', dest='batchSize', type=int, default=100,
                        help='Batch size for training and test sets.')
    parser.add_argument('--base', dest='base', type=str, default='./leNet5',
                        help='Base name of the network output and temp files.')
    parser.add_argument('--syn', dest='synapse', type=str, default=None,
                        help='Load from a previously saved network.')
    parser.add_argument('data', help='Directory or pkl.gz file for the ' +
                                     'training and test sets')
    options = parser.parse_args()

    # setup the logger
    log = setupLogging('cnnPreTrainer: ' + options.data, 
                       options.level, options.logfile)

    # NOTE: The pickleDataset will silently use previously created pickles if
    #       one exists (for efficiency). So watch out for stale pickles!
    chipArgs = {'chipSize': options.chipSize, 
                'numChips': options.numChips}
    train = ingestImagery(filepath=options.data, shared=True,
                          batchSize=options.batchSize, 
                          log=log, chipFunc=randomChip, kwargs=chipArgs)

    if options.synapse is not None :
        # load a previously saved network
        network = StackedAENetwork(train, log=log)
        network.load(options.synapse)
    else :
        network = buildStackedAENetwork(
            train, log=log,
            kernelConv=options.kernel, 
            kernelSizeConv=options.kernelSize, 
            downsampleConv=options.downsample, 
            learnConv=options.learnC, 
            momentumConv=[None]*len(options.kernel),
            dropoutConv=options.dropoutC,
            neuronFull=options.neuron, 
            learnFull=options.learnF, 
            momentumFull=[None]*len(options.neuron),
            dropoutFull=options.dropoutF)

    # train the SAE
    trainUnsupervised(network, __file__, options.data, 
                      numEpochs=options.limit, stop=options.stop, 
                      synapse=options.synapse, base=options.base, 
                      dropout=options.dropout, learnC=options.learnC,
                      learnF=options.learnF, contrF=options.contrF, 
                      kernel=options.kernel, neuron=options.neuron, log=log)

    # cleanup the network -- this ensures the profile is written
    del network