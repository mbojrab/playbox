import numpy as np
from theano import config, shared, dot, function
from theano.tensor.nnet import tanh, sigmoid

class ContiguousLayer() :
    '''This class describes a Contiguous Neural Layer.
       
       inputBuffer       : the input buffer for this layer
       inputPattern      : number of inputs to this layer
       numNeurons        : number of neurons in this layer
       initialWeights    : weights to initialize the network
                           None generates random weights for the layer
       initialThresholds : thresholds to initialize the network
                           None generates random thresholds for the layer
       activation        : the sigmoid function to use for activation
                           this must be a function with a derivative form
       randomNumGen      : generator for the initial weight values
    '''
    def __init__ (self, inputBuffer, inputPattern, numNeurons,
                  initialWeights=None, initialThresholds=None,
                  activation=tanh, randomNumGen=None) :

        self.input = inputBuffer
        self.inputPattern = inputPattern
        self.numNeurons = numNeurons
        
        # setup initial values for the weights
        if initialWeights is None :
            initialWeights = randomNumGen.uniform(
                low=-np.sqrt(6. / (self.inputPattern + self.numNeurons)),
                high=np.sqrt(6. / (self.inputPattern + self.numNeurons)),
                size=(self.inputPattern, self.numNeurons), dtype=config.floatX)
            if activation == sigmoid :
                self.weights *= 4
        self.weights = shared(value=initialWeights, borrow=True)

        # setup initial values for the thresholds
        if initialThresholds is None :
            initialThresholds = np.zeros((self.numNeurons,),
                                         dtype=config.floatX)
        self.thresholds = shared(value=initialThresholds, borrow=True)

        # the output buffer is now connected to a sequence of operations
        out = dot(self.input, self.weights) + self.thresholds
        self.output = (out if activation is None else activation(out))

        self.activate = function([self.input], self.output)

    def getInputSize (self) :
        return self.inputPattern
    def getOutputSize (self) :
        return self.numNeurons

    def activate (self, input) :
        self.activate(input)
        return self.output
