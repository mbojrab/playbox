import six
from nn.layer import Layer
from theano import dot
from theano.tensor import tanh

class ContiguousLayer(Layer) :
    '''This class describes a Contiguous Neural Layer.

       layerID           : unique name identifier for this layer
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
    def __init__ (self, layerID, inputSize, numNeurons,
                  learningRate=0.001, momentumRate=0.9, dropout=None,
                  initialWeights=None, initialThresholds=None, activation=tanh,
                  randomNumGen=None) :
        Layer.__init__(self, layerID, learningRate, momentumRate,
                       dropout, activation)

        self._inputSize = (None, inputSize[1])
        self._numNeurons = numNeurons

        # create weights based on the optimal distribution for the activation
        if initialWeights is None or initialThresholds is None :
            self._initializeWeights(
                size=(self._inputSize[1], self._numNeurons),
                fanIn=self._inputSize[1],
                fanOut=self._numNeurons,
                randomNumGen=randomNumGen)

    def finalize(self, networkInput, layerInput) :
        '''Setup the computation graph for this layer.
           networkInput : the input variable tuple for the network
                          format (inClass, inTrain)
           layerInput   : the input variable tuple for this layer
                          format (inClass, inTrain)
        '''
        self.input = layerInput

        # create the logits
        def findLogit(input, weights, thresholds) :
            # adjust the input for the correct number of dimensions
            input = input.flatten(2) if self.input[0].ndim > 2 else input
            return dot(input, weights) + thresholds
        outClass = findLogit(self.input[0], self._weights, self._thresholds)
        outTrain = findLogit(self.input[1], self._weights, self._thresholds)

        # create a convenience function
        self.output = self._setOutput((self._numNeurons,), outClass, outTrain)

    def getInputSize (self) :
        '''(numInputs, pattern size)'''
        return self._inputSize
    def getOutputSize (self) :
        '''(numInputs, number of neurons)'''
        return (self._inputSize[0], self._numNeurons)

    # DEBUG: For Debugging purposes only
    def writeWeights(self, ii, imageShape=None) :
        from dataset.debugger import saveTiledImage
        matSize = self._weights.get_value(borrow=True).shape

        buffer = self._weights.get_value(borrow=True).T
        if self.input[0].ndim > 2 :
            import numpy as np
            buffer = np.reshape(buffer, [self._numNeurons] +
                                        list(self.input[0].shape.eval()[1:]))
            imageShape = buffer.shape[-2:]

        # transpose the weight matrix to alighn the kernels contiguously
        saveTiledImage(
            image=buffer,
            path=self.layerID + '_cae_filters_' + str(ii) + '.png',
            imageShape=(1, matSize[0]) if imageShape is None else imageShape,
            tileShape=(matSize[1], 1) if imageShape is None else None,
            spacing=0 if imageShape is None else 1,
            interleave=True)
