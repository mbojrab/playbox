from layer import Layer
import numpy as np
from theano.tensor import tanh
from theano import shared, config, function
from theano.tensor import grad

class ConvolutionalLayer(Layer) :
    '''This class describes a Convolutional Neural Layer which specifies
       a series of kernels and subsample.

       layerID           : unique name identifier for this layer
       input             : the input buffer for this layer
       inputPattern      : (batch size, channels, rows, columns)
       kernelShape       : (number of kernels, channels, rows, columns)
       downsampleShape   : (rowFactor, columnFactor)
       learningRate      : learning rate for all neurons
       initialWeights    : weights to initialize the network
                           None generates random weights for the layer
       initialThresholds : thresholds to initialize the network
                           None generates random thresholds for the layer
       activation        : the sigmoid function to use for activation
                           this must be a function with a derivative form
       randomNumGen      : generator for the initial weight values
    '''
    def __init__ (self, layerID, input, inputPattern, kernelShape, 
                  downsampleShape, learningRate = 0.001, initialWeights=None, 
                  initialThresholds=None, activation=tanh, randomNumGen=None) :
        Layer.__init__(self, layerID, learningRate)

        if inputPattern[3] == kernelShape[3] or \
           inputPattern[4] == kernelShape[4] :
            raise ValueError('ConvolutionalLayer Error: ' +
                             'InputShape cannot equal filterShape')
        if inputPattern[1] != kernelShape[1] :
            raise ValueError('ConvolutionalLayer Error: ' +
                             'Number of Channels must match in ' +
                             'inputShape and kernelShape')
        from theano.tensor.nnet.conv import conv2d
        from theano.tensor.signal.downsample import max_pool_2d

        self.input = input
        self.inputPattern = inputPattern

        # setup initial values for the weights -- if necessary
        if initialWeights is None :
            # create a rng if its needed
            if randomNumGen is None :
               from numpy.random import RandomState
               randomNumGen = RandomState(1234)

            downRate = np.prod(downsampleShape)
            fanIn = np.prod(kernelShape[1:])
            fanOut = kernelShape[0] * np.prod(kernelShape[2:]) / downRate
            scaleFactor = np.sqrt(6. / (fanIn + fanOut))
            initialWeights = np.asarray(randomNumGen.uniform(
                    low=-scaleFactor, high=scaleFactor, size=kernelShape),
                    dtype=config.floatX)
        self._weights = shared(value=initialWeights, borrow=True)

        # setup initial values for the thresholds -- if necessary
        if initialThresholds is None :
            initialThresholds = np.zeros((kernelShape[0],),
                                         dtype=config.floatX)
        self._thresholds = shared(initialThresholds, borrow=True)

        # create a function to perform the convolution
        convolve = conv2d(self.input, self._weights,
                          self.inputPattern, kernelShape)

        # create a function to perform the max pooling
        pooling = max_pool_2d(convolve, downsampleShape, True)

        # the output buffer is now connected to a sequence of operations
        self.output = tanh(pooling + 
                           self._thresholds.dimshuffle('x', 0, 'x', 'x'))

        # we can call this method to activate the layer
        self.activate = function([self.input], self.output)

        # store off our sizing
        self.kernelShape = kernelShape
        self.downsampleShape = downsampleShape


    def getInputSize (self) :
        '''The initial input size provided at construction. This is sized
           (batch size, channels, rows, columns)'''
        return self.input.shape
    def getKernelSize (self) :
        '''The initial kernel size provided at construction. This is sized
           (number of kernels, channels, rows, columns)'''
        return self.kernelShape
    def getFeatureSize (self) :
        '''This is the post convolution size of the output.
           (channels, rows, columns)'''
        return (self.kernelShape[1],
                self.input.shape[2] - self.kernelShape[2] + 1,
                self.input.shape[3] - self.kernelShape[3] + 1)
    def getOutputSize (self) :
        '''This is the post downsample size of the output.
           (channels, rows, columns)'''
        fShape = self.getFeatureSize()
        return (fShape[0],
                fShape[1] / self.downsampleShape[1],
                fShape[2] / self.downsampleShape[2])

    def activate (self, input) :
        self.activate(input)
        return self.output

    def buildBackPropagate (self, cost) :
        self.grads = grad(cost, [self.W, self.b])