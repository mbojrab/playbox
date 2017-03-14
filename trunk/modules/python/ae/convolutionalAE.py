import numpy as np
from theano import config, shared, function
import theano.tensor as t
from ae.encoder import AutoEncoder
from nn.convolutionalLayer import ConvolutionalLayer
from theano.tensor.nnet.conv import conv2d

class ConvolutionalAutoEncoder(ConvolutionalLayer, AutoEncoder) :
    '''This class describes a Contractive AutoEncoder (CAE) for a convolutional
       layer. This differs from the normal CAE in that it is useful for
       contextual information like imagery or text.

       If the decoded message matches the original input, the encoders is
       considered lossless. Otherwise the loss is calculated and the encoder 
       is updated, so it can better encode the input when encountered again.
       Over time the object will extract regular patterns in the data which
       are frequently encountered.

       CAEs can be stacked and trained in a greedy layerwise manner, and the
       trained CAEs can be used to initialize a Neural Network into a better
       regularized state than random initialization. Lastly this technique can
       be used when the amount of unlabeled data far outweighs the amount of 
       labeled data.

       layerID          : unique name identifier for this layer
       inputSize        : (batch size, channels, rows, columns)
       kernelSize       : (number of kernels, channels, rows, columns)
       regType          : type of regularization term to use
                          default None : perform no additional regularization
                          L1           : Least Absolute Deviation
                          L2           : Least Squares
       downsampleFactor : (rowFactor, columnFactor)
       learningRate     : learning rate for all neurons
       momentumRate     : rate of momentum for all neurons
                          NOTE: momentum allows for higher learning rates
       contractionRate  : variance (dimensionality) reduction rate
                          None uses '1 / numNeurons'
       dropout          : rate of retention in a given neuron during training
                          NOTE: input layers should be around .8 or .9
                                hidden layers should be around .5 or .6
                                output layers should always be 1.
       initialWeights   : weights to initialize the network
                          None generates random weights for the layer
       initialHidThresh : thresholds to initialize the forward network
                          None generates random thresholds for the layer
       initialVisThresh : thresholds to initialize the backward network
                          None generates random thresholds for the layer
       activation       : the sigmoid function to use for activation
                          this must be a function with a derivative form
       forceSparsity    : round the output of the neurons to {0,1}
                          this put more emphasis on the pattern extraction
       randomNumGen     : generator for the initial weight values
    '''
    def __init__ (self, layerID, inputSize, kernelSize, 
                  downsampleFactor, regType=None,
                  learningRate=0.001, momentumRate=0.9, 
                  dropout=None, contractionRate=None,
                  initialWeights=None, initialHidThresh=None,
                  initialVisThresh=None, activation=t.nnet.sigmoid,
                  forceSparsity=True, randomNumGen=None) :
        from nn.reg import Regularization
        ConvolutionalLayer.__init__(self, layerID=layerID,
                                    inputSize=inputSize,
                                    kernelSize=kernelSize,
                                    downsampleFactor=downsampleFactor,
                                    learningRate=learningRate,
                                    momentumRate=momentumRate,
                                    dropout=dropout,
                                    initialWeights=initialWeights,
                                    initialThresholds=initialHidThresh,
                                    activation=activation, 
                                    randomNumGen=randomNumGen)
        AutoEncoder.__init__(self, forceSparsity, 
                             1. / np.prod(kernelSize[:]) \
                             if contractionRate is None else contractionRate)

        # setup initial values for the hidden thresholds
        if initialVisThresh is None :
            initialVisThresh = np.zeros((self._inputSize[1],),
                                        dtype=config.floatX)
        self._thresholdsBack = shared(value=initialVisThresh, borrow=True)
        self._regularization = Regularization(regType, self._contractionRate)

    def _setActivation(self, out) :
        from nn.layer import Layer
        from theano.tensor import round
        act = Layer._setActivation(self, out)
        return round(act, mode='half_away_from_zero') \
               if self._forceSparse else act

    def __getstate__(self) :
        '''Save layer pickle'''
        from dataset.shared import fromShared
        dict = ConvolutionalLayer.__getstate__(self)
        dict['_thresholdsBack'] = fromShared(self._thresholdsBack)
        # remove the functions -- they will be rebuilt JIT
        if 'reconstruction' in dict : del dict['reconstruction']
        if '_costs' in dict : del dict['_costs']
        if '_costLabels' in dict : del dict['_costLabels']
        if '_updates' in dict : del dict['_updates']
        if 'trainLayer' in dict : del dict['trainLayer']
        return dict

    def __setstate__(self, dict) :
        '''Load layer pickle'''
        from theano import shared
        # remove any current functions from the object so we force the
        # theano functions to be rebuilt with the new buffers
        if hasattr(self, 'reconstruction') : delattr(self, 'reconstruction')
        if hasattr(self, '_costs') : delattr(self, '_costs')
        if hasattr(self, '_costLabels') : delattr(self, '_costLabels')
        if hasattr(self, '_updates') : delattr(self, '_updates')
        if hasattr(self, 'trainLayer') : delattr(self, 'trainLayer')
        ConvolutionalLayer.__setstate__(self, dict)
        initialThresholdsBack = self._thresholdsBack
        self._thresholdsBack = shared(value=initialThresholdsBack, borrow=True)

    def _unpool_2d(self, input, upsampleFactor) :
        '''This method performs the opposite of pool_2d. This uses the index
           which produced the largest input during pooling in order to produce
           the sparse upsample.
        '''
        if input.ndim < len(self.getOutputSize()) :
            from dataset.shared import getShape
            input = input.reshape([getShape(input)[0]] +
                                  list(self.getOutputSize()[1:]))
        return input.repeat(upsampleFactor[0], axis=2).repeat(
                            upsampleFactor[1], axis=3) \
               if upsampleFactor[0] > 1 else input

    def _getWeightsBack(self) :
        '''Calculate the weights used for decoding.'''
        kernelSize = self.getKernelSize()
        kernelBackSize = (kernelSize[1], kernelSize[0], 
                          kernelSize[2], kernelSize[3])
        return t.reshape(self._weights, (kernelBackSize))

    def _decode(self, input) :
        from nn.layer import Layer
        weightsBack = self._getWeightsBack()
        deconvolve = conv2d(input, weightsBack, self.getFeatureSize(), 
                            weightsBack.shape.eval(), border_mode='full')
        out = deconvolve + self._thresholdsBack.dimshuffle('x', 0, 'x', 'x')
        return Layer._setActivation(self, out)

    def finalize(self, networkInput, layerInput) :
        '''Setup the computation graph for this layer.
           networkInput : the input variable tuple for the network
                          format (inClass, inTrain)
           layerInput   : the input variable tuple for this layer
                          format (inClass, inTrain)
        '''
        from nn.costUtils import calcLoss, leastSquares, \
                                 calcSparsityConstraint, compileUpdate
        from dataset.shared import getShape
        ConvolutionalLayer.finalize(self, networkInput, layerInput)

        weightsBack = self._getWeightsBack()
        self._costs = []
        self._costLabels = []

        # setup the decoder --
        # this take the output of the feedforward process as input and
        # and runs the output back through the network in reverse. The net
        # effect is to reconstruct the input, and ultimately to see how well
        # the network is at encoding the message.
        decodedInput = self.buildDecoder(self.output[0])

        # DEBUG: For Debugging purposes only
        self.reconstruction = function([networkInput[0]], decodedInput)

        # NOTE: Sparsity is not a useful constraint on convolutional layers

        # contraction is only applicable in the non-binary case 
        if not self._forceSparse :
            # compute the jacobian cost of the output --
            # This works as a sparsity constraint in case the hidden vector is
            # larger than the input vector.
            unpooling = self._unpool_2d(self.output[0], self._downsampleFactor)
            jacobianMat = conv2d(unpooling * (1. - unpooling), weightsBack,
                                 self.getFeatureSize(), weightsBack.shape.eval(),
                                 border_mode='full')
            self._costs.append(leastSquares(jacobianMat, self._contractionRate))
            self._costLabels.append('Jacob')

        # create the negative log likelihood function --
        # this is our cost function with respect to the original input
        # NOTE: The jacobian was computed however takes much longer to process
        #       and does not help convergence or regularization. It was removed
        self._costs.append(calcLoss(
            self.input[0], decodedInput, self._activation,
            scaleFactor=1. / self.getInputSize()[1]))
        self._costLabels.append('Local Cost')

        # add regularization if it was user requested
        regularization = self._regularization.calculate([self])
        if regularization is not None :
            self._costs.append(regularization)
            self._costLabels.append('Regularization')

        gradients = t.grad(t.sum(self._costs) / getShape(networkInput[0])[0],
                           self.getWeights())
        self._updates = compileUpdate(self.getWeights(), gradients,
                                      self._learningRate, self._momentumRate)

        # TODO: this needs to be stackable and take the input to the first
        #       layer, not just the input of this layer. This will ensure
        #       the other layers are activated to get the input to this layer
        # DEBUG: For Debugging purposes only
        self.trainLayer = function([networkInput[0]], self._costs,
                                   updates=self._updates)

    def buildDecoder(self, input) :
        '''Calculate the decoding component. This should be used after the
           encoder has been created. The decoder is ran in the opposite
           direction.
        '''
        # NOTE: the output may come back as a different shape than it left
        #       so we reshape here just in case.
        return self._decode(self._unpool_2d(input, self._downsampleFactor))

    def getWeights(self) :
        '''Update to account for the decode thresholds.'''
        return [self._weights, self._thresholds, self._thresholdsBack]

    def getUpdates(self) :
        '''This allows the Stacker to build the layerwise training.'''
        return (self._costs, self._updates)

    def getCostLabels(self) :
        '''Return the labels associated with the cost functions applied.'''
        return self._costLabels
