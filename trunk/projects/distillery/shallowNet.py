import theano.tensor as t
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

    input = t.ftensor4('input')

    trainSize = inputSize

    # create the network
    network = Net()

    # add convolutional layers
    network.addLayer(ConvolutionalLayer(
        layerID='c1', input=input, 
        inputSize=trainSize,
        kernelSize=(numKernels,trainSize[1],3,3),
        downsampleFactor=(3,3), randomNumGen=rng,
        learningRate=.08, momentumRate=.1))
    # add fully connected layers
    network.addLayer(ContiguousLayer(
        layerID='f2', input=network.getNetworkOutput(),
        inputSize=(network.getNetworkOutputSize()[0],
                   reduce(mul, network.getNetworkOutputSize()[1:])),
        numNeurons=numNeurons, randomNumGen=rng,
        learningRate=.025, momentumRate=.2))
    network.addLayer(ContiguousLayer(
        layerID='f3', input=network.getNetworkOutput(),
        inputSize=network.getNetworkOutputSize(), numNeurons=numLabels,
        learningRate=.015, momentumRate=.3, activation=None, randomNumGen=rng))

    return network
    

'''This is a simple network in the topology of leNet5 the well-known
   MNIST dataset trainer from Yann LeCun. This is capable of training other
   datasets, however the sizing must be correct.
'''
if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    parser.add_argument('--size', dest='inputSize', type=list, nargs=4,
                        default=[100,1,28,28], help='Dimensions of the input.')
    parser.add_argument('--numLabels', dest='labels', type=int, default=10, 
                        help='Number of neurons in the output layer.')
    parser.add_argument('--out', dest='out', type=str,
                        default='./shallowNet_3Layers_1conv_2fc.pkl.gz',
                        help='Base name of the network output and temp files.')
    options = parser.parse_args()
    network = createNetwork(inputSize=options.inputSize, numKernels=50, 
                            numNeurons=100, numLabels=options.labels)

    # save to the output file
    network.save(options.out)
