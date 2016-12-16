import numpy as np
from theano import config, shared, dot, function
import theano.tensor as t
from ae.encoder import AutoEncoder
from nn.contiguousLayer import ContiguousLayer

class ContiguousAutoEncoder(ContiguousLayer, AutoEncoder) :
    '''This class describes a Contractive AutoEncoder (CAE) for a Contiguous
       Layer. This is a unsupervised learning technique which encodes an input
       (feedforward), and then decodes the output vector by sending backwards
       through the layer. This attempts to reconstruct the original input. 

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
       inputSize        : number of elements in input buffer
       numNeurons       : number of neurons in this layer
       regType          : type of regularization term to use
                          default None : perform no additional regularization
                          L1           : Least Absolute Deviation
                          L2           : Least Squares
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
    def __init__ (self, layerID, inputSize, numNeurons, regType=None,
                  learningRate=0.001, momentumRate=0.9, 
                  dropout=None, contractionRate=None,
                  initialWeights=None, initialHidThresh=None,
                  initialVisThresh=None, activation=t.nnet.sigmoid,
                  forceSparsity=True, randomNumGen=None) :
        from nn.reg import Regularization
        ContiguousLayer.__init__(self, layerID=layerID,
                                 inputSize=inputSize,
                                 numNeurons=numNeurons,
                                 learningRate=learningRate,
                                 momentumRate=momentumRate,
                                 dropout=dropout,
                                 initialWeights=initialWeights,
                                 initialThresholds=initialHidThresh,
                                 activation=activation, 
                                 randomNumGen=randomNumGen)
        AutoEncoder.__init__(self, forceSparsity,
                             1. / numNeurons if contractionRate is None \
                             else contractionRate)
        self._regularization = Regularization(regType, self._contractionRate)

        # setup initial values for the hidden thresholds
        if initialVisThresh is None :
            initialVisThresh = np.zeros((self._inputSize[1],), 
                                        dtype=config.floatX)
        self._thresholdsBack = shared(value=initialVisThresh, borrow=True)

    def _setActivation(self, out) :
        from nn.layer import Layer
        from theano.tensor import round
        act = Layer._setActivation(self, out)
        return round(act) if self._forceSparse else act

    def __getstate__(self) :
        '''Save layer pickle'''
        from dataset.shared import fromShared
        dict = ContiguousLayer.__getstate__(self)
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
        ContiguousLayer.__setstate__(self, dict)
        initialThresholdsBack = self._thresholdsBack
        self._thresholdsBack = shared(value=initialThresholdsBack, borrow=True)

    def _decode(self, output) :
        from nn.layer import Layer
        out = dot(output, self._weights.T) + self._thresholdsBack
        return Layer._setActivation(self, out)

    def finalize(self, networkInput, layerInput) :
        '''Setup the computation graph for this layer.
           networkInput : the input variable tuple for the network
                          format (inClass, inTrain)
           layerInput   : the input variable tuple for this layer
                          format (inClass, inTrain)
        '''
        from nn.costUtils import calcLoss, computeJacobian, leastSquares, \
                                 calcSparsityConstraint, compileUpdate
        ContiguousLayer.finalize(self, networkInput, layerInput)

        # setup the decoder --
        # this take the output of the feedforward process as input and
        # and runs the output back through the network in reverse. The net
        # effect is to reconstruct the input, and ultimately to see how well
        # the network is at encoding the message.
        decodedInput = self._decode(self.output[0])
        self._costs = []
        self._costLabels = []

        # DEBUG: For Debugging purposes only
        self.reconstruction = function([networkInput[0]], decodedInput)
        self._costs.append(calcSparsityConstraint(
            self.output[0], self.getOutputSize()))
        self._costLabels.append('Sparsity')

        # contraction is only applicable in the non-binary case 
        if not self._forceSparse :
            # compute the jacobian cost of the output --
            # This works as a sparsity constraint in case the hidden vector is
            # larger than the input vector.
            self._costs.append(leastSquares(
                computeJacobian(self.output[0], self._weights,
                                self._inputSize[0], self._inputSize[1],
                                self._numNeurons), 
                self._inputSize[0], self._contractionRate))
            self._costLabels.append('Jacob')

        # create the negative log likelihood function --
        # this is our cost function with respect to the original input
        self._costs.append(calcLoss(self.input[0].flatten(2), decodedInput,
                           self._activation))
        self._costLabels.append('Local Cost')

        # add regularization if it was user requested
        regularization = self._regularization.calculate([self])
        if regularization is not None :
            self._costs.append(regularization)
            self._costLabels.append('Regularization')

        gradients = t.grad(t.sum(self._costs) / self.getInputSize()[0],
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
        return self._decode(input)

    def getWeights(self) :
        '''Update to account for the decode thresholds.'''
        return [self._weights, self._thresholds, self._thresholdsBack]

    def getUpdates(self) :
        '''This allows the Stacker to build the layerwise training.'''
        return (self._costs, self._updates)

    def getCostLabels(self) :
        '''Return the labels associated with the cost functions applied.'''
        return self._costLabels
