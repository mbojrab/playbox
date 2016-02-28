from layer import Layer
import numpy as np
from theano import config, shared, dot, function
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

class ContiguousLayer(Layer) :
    '''This class describes a Contiguous Neural Layer.
       
       layerID           : unique name identifier for this layer
       input             : the input buffer for this layer
       inputSize         : number of elements in input buffer
       numNeurons        : number of neurons in this layer
       learningRate      : learning rate for all neurons
       initialWeights    : weights to initialize the network
                           None generates random weights for the layer
       initialThresholds : thresholds to initialize the network
                           None generates random thresholds for the layer
       activation        : the sigmoid function to use for activation
                           this must be a function with a derivative form
       randomNumGen      : generator for the initial weight values
    '''
    def __init__ (self, layerID, input, inputSize, numNeurons,
                  learningRate=0.001, initialWeights=None,
                  initialThresholds=None, activation=tanh, randomNumGen=None) :
        Layer.__init__(self, layerID, learningRate)

        # store the input buffer
        self.input = input
        self._inputSize = inputSize
        if isinstance(self._inputSize, long) or len(self._inputSize) is not 2 :
            self._inputSize = (1, inputSize)
        self._numNeurons = numNeurons

        # setup initial values for the weights
        if initialWeights is None :
            # create a rng if its needed
            if randomNumGen is None :
               from numpy.random import RandomState
               from time import time
               randomNumGen = RandomState(int(time()))

            initialWeights = np.asarray(randomNumGen.uniform(
                low=-np.sqrt(6. / (self._inputSize[1] + self._numNeurons)),
                high=np.sqrt(6. / (self._inputSize[1] + self._numNeurons)),
                size=(self._inputSize[1], self._numNeurons)),
                dtype=config.floatX)
            if activation == sigmoid :
                initialWeights *= 4.
        self._weights = shared(value=initialWeights, borrow=True)

        # setup initial values for the thresholds
        if initialThresholds is None :
            initialThresholds = np.zeros((self._numNeurons,),
                                         dtype=config.floatX)
        self._thresholds = shared(value=initialThresholds, borrow=True)

        out = dot(self.input, self._weights) + self._thresholds
        self.output = out if activation is None else activation(out)
        self.activate = function([self.input], self.output)

    def getWeights(self) :
        '''This allows the network backprop all layers efficiently.'''
        return [self._weights, self._thresholds]
    def getInputSize (self) :
        '''(numInputs, pattern size)'''
        return self.inputSize
    def getOutputSize (self) :
        '''(numInputs, number of neurons)'''
        return (self.inputSize[0], self.numNeurons)
