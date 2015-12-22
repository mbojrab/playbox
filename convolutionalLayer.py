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
       inputSize         : (batch size, channels, rows, columns)
       kernelSize        : (number of kernels, channels, rows, columns)
       downsampleFactor  : (rowFactor, columnFactor)
       learningRate      : learning rate for all neurons
       initialWeights    : weights to initialize the network
                           None generates random weights for the layer
       initialThresholds : thresholds to initialize the network
                           None generates random thresholds for the layer
       activation        : the sigmoid function to use for activation
                           this must be a function with a derivative form
       randomNumGen      : generator for the initial weight values
    '''
    def __init__ (self, layerID, input, inputSize, kernelSize, 
                  downsampleFactor, learningRate=0.001, initialWeights=None, 
                  initialThresholds=None, activation=tanh, randomNumGen=None) :
        Layer.__init__(self, layerID, learningRate)

        # TODO: this check is likely unnecessary
        if inputSize[2] == kernelSize[2] or inputSize[3] == kernelSize[3] :
            raise ValueError('ConvolutionalLayer Error: ' +
                             'inputSize cannot equal kernelSize')
        if inputSize[1] != kernelSize[1] :
            raise ValueError('ConvolutionalLayer Error: ' +
                             'Number of Channels must match in ' +
                             'inputSize and kernelSize')
        from theano.tensor.nnet.conv import conv2d
        from theano.tensor.signal.downsample import max_pool_2d

        # theano variables don't actually preserve buffer sizing
        self.input = input
        self.inputSize = inputSize
        self.kernelSize = kernelSize
        self.downsampleFactor = downsampleFactor

        # setup initial values for the weights -- if necessary
        if initialWeights is None :
            # create a rng if its needed
            if randomNumGen is None :
               from numpy.random import RandomState
               randomNumGen = RandomState(1234)

            # this creates optimal initial weights by randomizing them
            # to an appropriate range around zero, which leads to better
            # convergence.
            downRate = np.prod(self.downsampleFactor)
            fanIn = np.prod(self.kernelSize[1:])
            fanOut = self.kernelSize[0] * \
                     np.prod(self.kernelSize[2:]) / downRate
            scaleFactor = np.sqrt(6. / (fanIn + fanOut))
            initialWeights = np.asarray(randomNumGen.uniform(
                    low=-scaleFactor, high=scaleFactor, size=self.kernelSize),
                    dtype=config.floatX)
        self._weights = shared(value=initialWeights, borrow=True)

        # setup initial values for the thresholds -- if necessary
        if initialThresholds is None :
            initialThresholds = np.zeros((self.kernelSize[0],),
                                         dtype=config.floatX)
        self._thresholds = shared(value=initialThresholds, borrow=True)

        # create a function to perform the convolution
        convolve = conv2d(self.input, self._weights,
                          self.inputSize, self.kernelSize)

        # create a function to perform the max pooling
        pooling = max_pool_2d(convolve, self.downsampleFactor, True)

        # the output buffer is now connected to a sequence of operations
        self.output = tanh(pooling + 
                           self._thresholds.dimshuffle('x', 0, 'x', 'x'))

        # we can call this method to activate the layer
        self.activate = function([self.input], self.output)

    def getWeights(self) :
        '''This allows the network backprop all layers efficiently.'''
        return [self._weights, self._thresholds]
    def getInputSize (self) :
        '''The initial input size provided at construction. This is sized
           (batch size, channels, rows, columns)'''
        return self.inputSize
    def getKernelSize (self) :
        '''The initial kernel size provided at construction. This is sized
           (number of kernels, channels, rows, columns)'''
        return self.kernelSize
    def getFeatureSize (self) :
        '''This is the post convolution size of the output.
           (channels, rows, columns)'''
        return (self.inputSize[0], self.kernelSize[0],
                self.inputSize[2] - self.kernelSize[2] + 1,
                self.inputSize[3] - self.kernelSize[3] + 1)
    def getOutputSize (self) :
        '''This is the post downsample size of the output.
           (numImages, channels, rows, columns)'''
        fShape = self.getFeatureSize()
        return (fShape[0], fShape[1],
                fShape[2] / self.downsampleFactor[0],
                fShape[3] / self.downsampleFactor[1])

    def activate (self, input) :
        self.activate(input)
        return self.output

    def buildBackPropagate (self, cost) :
        self.grads = grad(cost, [self.W, self.b])