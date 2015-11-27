import theano
import theano.tensor as t
from net import Net
from contiguousLayer import ContiguousLayer
from convolutionalLayer import ConvolutionalLayer

if __name__ == '__main__' :

    # create a random number generator for efficiency
    from numpy.random import RandomState
    from time import time
    rng = RandomState(int(time()))

    input = t.fvector('input')
    expectedOutput = t.bvector('expectedOutput')

    # create the network -- LeNet-5
    network = Net(regType='', learningRate=.01, runCPU=False)

    # add convolutional layers
    network.addLayer(ConvolutionalLayer('c1', input, (1,1,28,28), (6,5,5),
                                        (2,2), runCPU=True, randomNumGen=rng))
    network.addLayer(ConvolutionalLayer('c2', network.getNetworkOutput(),
                                        network.getOutputSize(), (6,5,5),
                                        (2,2), runCPU=True, randomNumGen=rng))

    # add fully connected layers
    network.addLayer(
        ContiguousLayer('f3', network.getNetworkOutput().flatten(2), 5, 3))
    network.addLayer(
        ContiguousLayer('f4', network.getNetworkOutput(), 3, 3))

    from time import time
    numRuns = 10000
    arr = [1, 2, 3, 4, 5]
    exp = [-1, -1, 1]

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
