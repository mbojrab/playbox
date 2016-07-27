import argparse
from time import time
from six.moves import reduce

from nn.net import ClassifierNetwork as Net
from nn.contiguousLayer import ContiguousLayer
from nn.convolutionalLayer import ConvolutionalLayer


def createNetwork(inputSize, numKernels, numNeurons, numLabels) :

    # create a random number generator for efficiency
    from numpy.random import RandomState
    from operator import mul
    rng = RandomState(int(time()))

    trainSize = inputSize

    # create the network
    network = Net()

    # add convolutional layers
    network.addLayer(ConvolutionalLayer(
        layerID='c1', inputSize=trainSize,
        kernelSize=(numKernels,trainSize[1],7,7),
        downsampleFactor=(2,2), randomNumGen=rng,
        learningRate=.09, momentumRate=.9))
    # add fully connected layers
    network.addLayer(ContiguousLayer(
        layerID='f2', 
        inputSize=(network.getNetworkOutputSize()[0],
                   reduce(mul, network.getNetworkOutputSize()[1:])),
        numNeurons=numNeurons, randomNumGen=rng,
        learningRate=.03, momentumRate=.7))
    network.addLayer(ContiguousLayer(
        layerID='f3', inputSize=network.getNetworkOutputSize(), 
        numNeurons=numLabels, learningRate=.01, momentumRate=.7, 
        activation=None, randomNumGen=rng))

    return network


'''This is a simple network in the topology of leNet5 the well-known
   MNIST dataset trainer from Yann LeCun. This is capable of training other
   datasets, however the sizing must be correct.
'''
if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    parser.add_argument('--size', dest='inputSize', type=list, nargs=4,
                        default=[50,1,28,28], help='Dimensions of the input.')
    parser.add_argument('--numLabels', dest='labels', type=int, default=10, 
                        help='Number of neurons in the output layer.')
    parser.add_argument('--out', dest='out', type=str,
                        default='./shallowNet_3Layers_1conv_2fc.pkl.gz',
                        help='Base name of the network output and temp files.')
    options = parser.parse_args()
    network = createNetwork(inputSize=options.inputSize, numKernels=8, 
                            numNeurons=30, numLabels=options.labels)

    # save to the output file
    network.save(options.out)
