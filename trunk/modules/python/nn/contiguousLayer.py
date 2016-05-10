from layer import Layer
import numpy as np
from theano import config, shared, dot, function
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh, switch

class ContiguousLayer(Layer) :
    '''This class describes a Contiguous Neural Layer.
       
       layerID           : unique name identifier for this layer
       input             : the input buffer for this layer
       inputSize         : number of elements in input buffer. This can also
                           be a tuple of size (batch size, input vector length)
       numNeurons        : number of neurons in this layer
       learningRate      : learning rate for all neurons
       momentumRate      : rate of momentum for all neurons
                           NOTE: momentum allows for higher learning rates
       dropout           : rate of retention in a given neuron during training
                           NOTE: input layers should be around .8 or .9
                                 hidden layers should be around .5 or .6
                                 output layers should always be 1.
       initialWeights    : weights to initialize the network
                           None generates random weights for the layer
       initialThresholds : thresholds to initialize the network
                           None generates random thresholds for the layer
       activation        : the sigmoid function to use for activation
                           this must be a function with a derivative form
       randomNumGen      : generator for the initial weight values - 
                           type is numpy.random.RandomState
    '''
    def __init__ (self, layerID, input, inputSize, numNeurons,
                  learningRate=0.001, momentumRate=0.9, dropout=None,
                  initialWeights=None, initialThresholds=None, activation=tanh,
                  randomNumGen=None) :
        Layer.__init__(self, layerID, learningRate, momentumRate, dropout)

        # adjust the input for the correct number of dimensions        
        if isinstance(input, tuple) :
            if input[1].ndim > 2 : input = input[0].flatten(2), \
                                           input[1].flatten(2)
        else :
            if input.ndim > 2 : input = input.flatten(2)

        # store the input buffer -- this can either be a tuple or scalar
        # The input layer will only have a scalar so its duplicated here
        self.input = input if isinstance(input, tuple) else (input, input)
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

        # create the logits
        def findLogit(input, weights, thresholds) :
            return dot(input, weights) + thresholds
        outClass = findLogit(self.input[0], self._weights, self._thresholds)
        outTrain = findLogit(self.input[1], self._weights, self._thresholds)

        # determine dropout if requested
        if self._dropout is not None :
            # here there are two possible paths --
            # outClass : path of execution intended for classification. Here
            #            all neurons are present and weights must be scaled by
            #            the dropout factor. This ensures resultant 
            #            probabilities fall within intended bounds when all
            #            neurons are present.
            # outTrain : path of execution for training with dropout. Here each
            #            neuron's output goes through a Bernoulli Trial. This
            #            retains a neuron with the probability specified by the
            #            dropout factor.
            outClass = outClass / self._dropout
            outTrain = switch(self._randStream.binomial(
                size=(self._numNeurons,), p=self._dropout), outTrain, 0)

        # activate the layer --
        # output is a tuple to represent two possible paths through the
        # computation graph. 
        self.output = (outClass, outTrain) if activation is None else \
                      (activation(outClass), activation(outTrain))

        # create a convenience function
        self.activate = function([self.input[0]], self.output[0])

    def getWeights(self) :
        '''This allows the network backprop all layers efficiently.'''
        return [self._weights, self._thresholds]
    def getInputSize (self) :
        '''(numInputs, pattern size)'''
        return self._inputSize
    def getOutputSize (self) :
        '''(numInputs, number of neurons)'''
        return (self._inputSize[0], self._numNeurons)

    # DEBUG: For Debugging purposes only
    def writeWeights(self, ii, imageShape=None) :
        from nn.debugger import saveTiledImage
        matSize = self._weights.get_value(borrow=True).shape

        # transpose the weight matrix to alighn the kernels contiguously
        saveTiledImage(
            image=self._weights.get_value(borrow=True).T,
            path=self.layerID + '_cae_filters_' + str(ii) + '.png',
            imageShape=(1, matSize[0]) if imageShape is None else imageShape,
            tileShape=(matSize[1], 1) if imageShape is None else None,
            spacing=0 if imageShape is None else 1,
            interleave=True)
