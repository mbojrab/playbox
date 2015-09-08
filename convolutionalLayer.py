from exception import NotImplementedError, ValueError
import numpy as np
from theano.tensor.nnet import tanh
from theano import shared, config, function

class ConvolutionalLayer() :
    '''This class describes a Convolutional Neural Layer which specifies
       a series of kernels and subsample.
       
       inputBuffer       : the input buffer for this layer
       inputShape        : (batch size, channels, rows, columns)
       kernelShape       : (number of kernels, channels, rows, columns)
       downsampleShape   : (rowFactor, columnFactor)
       initialWeights    : weights to initialize the network
                           None generates random weights for the layer
       initialThresholds : thresholds to initialize the network
                           None generates random thresholds for the layer
       activation        : the sigmoid function to use for activation
                           this must be a function with a derivative form
       randomNumGen      : generator for the initial weight values
    '''
    def __init__ (self, inputBuffer, inputShape, kernelShape, downsampleShape,
                  initialWeights=None, initialThresholds=None, 
                  activation=tanh, randomNumGen=None) :
        if inputShape[3] == kernelShape[3] or inputShape[4] == kernelShape[4] :
            raise ValueError('ConvolutionalLayer Error: ' +
                             'InputShape cannot equal filterShape')
        if inputShape[1] != kernelShape[1] :
            raise ValueError('ConvolutionalLayer Error: ' +
                             'Number of Channels must match in ' +
                             'inputShape and kernelShape')
        from theano.tensor.nnet.conv import conv2d
        from theano.tensor.signal.downsample import max_pool_2d

        self.input = inputBuffer

        # setup initial values for the weights
        if initialWeights is None :
            downRate = np.prod(downsampleShape)
            fanIn = np.prod(kernelShape[1:])
            fanOut = kernelShape[0] * np.prod(kernelShape[2:]) / downRate
            scaleFactor = np.sqrt(6. / (fanIn + fanOut))
            initialWeights = np.asarray(randomNumGen.uniform(low=-scaleFactor,
                                                             high=scaleFactor,
                                                             size=kernelShape),
                                        dtype=config.floatX)
        self.weights = shared(value=initialWeights, borrow=True)

        # setup initial values for the thresholds
        if initialThresholds is None :
            initialThresholds = np.zeros((kernelShape[0],),
                                         dtype=config.floatX)
        self.thresholds = shared(initialThresholds, borrow=True)

        # create a function to perform the convolution
        convolve = conv2d(self.input, self.weights, inputBuffer, kernelShape)

        # create a function to perform the max pooling
        pooling = max_pool_2d(convolve, downsampleShape, True)

        # the output buffer is now connected to a sequence of operations
        self.output = tanh(pooling + 
                           self.thresholds.dimshuffle('x', 0, 'x', 'x'))

        # we can call this method to activate the layer
        self.activate = function([self.input], self.output)
        
        # store off our sizing
        self.inShape = inputBuffer
        self.kShape = kernelShape
        self.dShape = downsampleShape


    def getInputSize (self) :
        '''The initial input size provided at construction. This is sized
           (batch size, channels, rows, columns)'''
        return self.inShape
    def getKernelSize (self) :
        '''The initial kernel size provided at construction. This is sized
           (number of kernels, channels, rows, columns)'''
        return self.kShape
    def getFeatureSize (self) :
        '''This is the post convolution size of the output.'''
        return (self.kShape[1],
                self.inShape[2] - self.kShape[2] + 1,
                self.inShape[3] - self.kShape[3] + 1)
    def getOutputSize (self) :
        '''This is the post downsample size of the output.'''
        fShape = self.getFeatureSize()
        return (fShape[0],
                fShape[1] / self.dShape[1],
                fShape[2] / self.dShape[2])

    def activate (self, input) :
        self.activate(input)
        return self.output
