from layer import Layer
import numpy as np
from theano import config, shared, dot, function
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

class ContiguousLayer(Layer) :
    '''This class describes a Contiguous Neural Layer.
       
       layerID           : unique name identifier for this layer
       input             : the input buffer for this layer
       inputPattern      : number of elements in input buffer
       numNeurons        : number of neurons in this layer
       learningRate      : learning rate for all neurons
       initialWeights    : weights to initialize the network
                           None generates random weights for the layer
       initialThresholds : thresholds to initialize the network
                           None generates random thresholds for the layer
       activation        : the sigmoid function to use for activation
                           this must be a function with a derivative form
       runCPU            : run processing on CPU
       randomNumGen      : generator for the initial weight values
    '''
    def __init__ (self, layerID, input, inputPattern, numNeurons,
                  learningRate = .001, initialWeights=None,
                  initialThresholds=None, activation=tanh, runCPU=True,
                  randomNumGen=None) :
        Layer.__init__(self, layerID, runCPU)

        # store the input buffer
        self.input = input
        self.inputPattern = inputPattern
        self._learningRate = learningRate

        # setup initial values for the weights
        if initialWeights is None :
            # create a rng if its needed
            if randomNumGen is None :
               from numpy.random import RandomState
               randomNumGen = RandomState(1234)

            initialWeights = np.asarray(randomNumGen.uniform(
                low=-np.sqrt(6. / (self.inputPattern + numNeurons)),
                high=np.sqrt(6. / (self.inputPattern + numNeurons)),
                size=(self.inputPattern, numNeurons)),
                dtype=config.floatX)
            if activation == sigmoid :
                self.weights *= 4.
        self._weights = shared(value=initialWeights, borrow=True)

        # setup initial values for the thresholds
        if initialThresholds is None :
            initialThresholds = np.zeros((numNeurons,),
                                         dtype=config.floatX)
        self._thresholds = shared(value=initialThresholds, borrow=True)

        out = dot(self.input, self._weights) + self._thresholds
        self.output = out if activation is None else activation(out)
        self.activation = function([self.input], self.output)

    def getWeights(self) :
        '''This allows the network backprop all layers efficiently.'''
        return [self._weights, self._thresholds]
    def getInputSize (self) :
        '''(1, pattern size)'''
        return self.input.shape
    def getOutputSize (self) :
        '''(1, number of neurons)'''
        return self.output.shape
    def activate (self, input) :
        '''activate the neural layer'''
        return self.activation(input)
    def backPropagate (self, cost) :
        '''Update the weights and thresholds. The grad() method will calculate
           the error gradient automatically with respect to these weights from
           the final output. No need to keep track of this for each layer.
        '''
        from theano.tensor import grad
        self._weights -= self._learningRate * grad(cost, self._weights)
        self._thresholds -= self._learningRate * grad(cost, self._thresholds)

if __name__ == '__main__' :
    from theano import tensor
    x = tensor.fmatrix('x')
    y = tensor.fvector('y')

    imageSize = (28,28)
    inputPattern = np.prod(imageSize)
    layer = ContiguousLayer('layer0', x, inputPattern, 10, .001)

    from numpy.random import RandomState
    randomNumGen = RandomState(1234)
    x1 = randomNumGen.uniform(low=-1., high=1.,
                              size=(1, inputPattern)).astype(config.floatX)

    # time the activation
    from time import time
    t = time()
    for i in range(100000) :
        out = layer.activate(x1)
    print "total time: " + str(time() - t) + "s"


    # time the training
    t = time()
    for i in range(100000) :
        layer.activate(x1)
    print "total time: " + str(time() - t) + "s"
