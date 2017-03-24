from nn.layer import Layer
import numpy as np
from theano.tensor import tanh

class ConvolutionalLayer(Layer) :
    '''This class describes a Convolutional Neural Layer which specifies
       a series of kernels and subsample.

       layerID           : unique name identifier for this layer
       inputSize         : (batch size, channels, rows, columns)
       kernelSize        : (number of kernels, channels, rows, columns)
       downsampleFactor  : (rowFactor, columnFactor)
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
    def __init__ (self, layerID, inputSize, kernelSize,
                  downsampleFactor, learningRate=0.001, momentumRate=0.9,
                  dropout=None, initialWeights=None, initialThresholds=None,
                  activation=tanh, randomNumGen=None) :
        Layer.__init__(self, layerID, learningRate, momentumRate, dropout,
                       activation)

        if inputSize[1] != kernelSize[1] :
            raise ValueError('ConvolutionalLayer Error: ' +
                             'Number of Channels must match in ' +
                             'inputSize and kernelSize')

        # NOTE: use None instead of the batch size to allow variable batch
        #       sizes during deployment.
        self._inputSize = [None] + list(inputSize[1:])
        self._kernelSize = kernelSize
        self._downsampleFactor = downsampleFactor

        # create weights based on the optimal distribution for the activation
        if initialWeights is None or initialThresholds is None :
            self._initializeWeights(
                size=self._kernelSize,
                fanIn=np.prod(self._inputSize[1:]),
                fanOut=self._kernelSize[0],
                randomNumGen=randomNumGen)

    def finalize(self, networkInput, layerInput) :
        '''Setup the computation graph for this layer.
           networkInput : the input variable tuple for the network
                          format (inClass, inTrain)
           layerInput   : the input variable tuple for this layer
                          format (inClass, inTrain)
        '''
        self.input = layerInput

        def findLogits(input, weights, inputSize, kernelSize,
                       downsampleFactor, thresholds) :
            from theano.tensor.nnet.conv import conv2d
            from theano.tensor.signal.pool import pool_2d

            # create a function to perform the convolution
            convolve = conv2d(input, weights, inputSize, kernelSize)

            # create a function to perform the max pooling
            pooling = pool_2d(convolve, downsampleFactor, ignore_border=True)

            # the output buffer is now connected to a sequence of operations
            return pooling + thresholds.dimshuffle('x', 0, 'x', 'x'), convolve

        outClass, convClass = findLogits(self.input[0], self._weights,
                                         self._inputSize, self._kernelSize,
                                         self._downsampleFactor,
                                         self._thresholds)
        outTrain, convTrain = findLogits(self.input[1], self._weights,
                                         self._inputSize, self._kernelSize,
                                         self._downsampleFactor,
                                         self._thresholds)

        # create a convenience function
        self.output = self._setOutput(self.getOutputSize()[1:],
                                      outClass, outTrain)

    def getInputSize (self) :
        '''The initial input size provided at construction. This is sized
           (batch size, channels, rows, columns)'''
        return self._inputSize
    def getKernelSize (self) :
        '''The initial kernel size provided at construction. This is sized
           (number of kernels, channels, rows, columns)'''
        return self._kernelSize
    def getFeatureSize (self) :
        '''This is the post convolution size of the output.
           (batch size, number of kernels, rows, columns)'''
        return (self._inputSize[0],
                self._kernelSize[0],
                self._inputSize[2] - self._kernelSize[2] + 1,
                self._inputSize[3] - self._kernelSize[3] + 1)
    def getOutputSize (self) :
        '''This is the post downsample size of the output.
           (batch size, number of kernels, rows, columns)'''
        fShape = self.getFeatureSize()
        return (fShape[0], fShape[1],
                int(fShape[2] / self._downsampleFactor[0]),
                int(fShape[3] / self._downsampleFactor[1]))

    # DEBUG: For Debugging purposes only
    def writeWeights(self, ii) :
        from dataset.debugger import saveTiledImage
        saveTiledImage(image=self._weights.get_value(borrow=True),
                       path=self.layerID + '_cae_filters_' + str(ii) + '.png',
                       imageShape=(self._kernelSize[2], self._kernelSize[3]),
                       spacing=1,
                       interleave=True)
