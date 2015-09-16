from layer import Layer
import numpy as np
from theano import config, shared, dot, function
from theano.tensor.nnet import tanh, sigmoid

class ContiguousLayer(Layer) :
    '''This class describes a Contiguous Neural Layer.
       
       input             : the input buffer for this layer
       numNeurons        : number of neurons in this layer
       initialWeights    : weights to initialize the network
                           None generates random weights for the layer
       initialThresholds : thresholds to initialize the network
                           None generates random thresholds for the layer
       activation        : the sigmoid function to use for activation
                           this must be a function with a derivative form
       runCPU            : run processing on CPU
       randomNumGen      : generator for the initial weight values
    '''
    def __init__ (self, layerID, input, numNeurons,
                  initialWeights=None, initialThresholds=None,
                  activation=tanh, runCPU=True, randomNumGen=None) :
        Layer.__init__(self, layerID, runCPU)

        # store the input buffer
        self.input = input

        # setup initial values for the weights
        if initialWeights is None :
            initialWeights = randomNumGen.uniform(
                low=-np.sqrt(6. / (self.input.size() + numNeurons)),
                high=np.sqrt(6. / (self.input.size() + numNeurons)),
                size=(self.inputPattern, numNeurons), dtype=config.floatX)
            if activation == sigmoid :
                self.weights *= 4
        self._weights = shared(value=initialWeights, borrow=True)

        # setup initial values for the thresholds
        if initialThresholds is None :
            initialThresholds = np.zeros((self.numNeurons,),
                                         dtype=config.floatX)
        self._thresholds = shared(value=initialThresholds, borrow=True)

        # the output buffer is now connected to a sequence of operations
        out = dot(self.input, self.weights) + self.thresholds
        self.output = (out if activation is None else activation(out))

        self.activate = function([self.input], self.output)

    def activate (self, input) :
        self.activate(input)
        return self.output

#    def backPropagate (input, errorGrad, backError) :


    def getInputSize (self) :
        '''(pattern size,)'''
        return self.input.shape

    def getOutputSize (self) :
        '''(number of neurons,)'''
        return self.output.shape
