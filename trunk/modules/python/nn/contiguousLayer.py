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

        self._inputSize = inputSize
        if isinstance(self._inputSize, six.integer_types) or \
           len(self._inputSize) is not 2 :
            self._inputSize = (1, inputSize)
        self._numNeurons = numNeurons


        # create weights based on the optimal distribution for the activation
        if initialWeights is None or initialThresholds is None :
            self._initializeWeights(
                size=(self._inputSize[1], self._numNeurons), 
                fanIn=self._inputSize[1],
                fanOut=self._numNeurons,
                randomNumGen=randomNumGen)

    def finalize(self, input) :
        '''Setup the computation graph for this layer.
           input : the input variable tuple for this layer
                   format (inClass, inTrain)
        '''
        self.input = input

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

        # transpose the weight matrix to alighn the kernels contiguously
        saveTiledImage(
            image=self._weights.get_value(borrow=True).T,
            path=self.layerID + '_cae_filters_' + str(ii) + '.png',
            imageShape=(1, matSize[0]) if imageShape is None else imageShape,
            tileShape=(matSize[1], 1) if imageShape is None else None,
            spacing=0 if imageShape is None else 1,
            interleave=True)
