from nn.contiguousLayer import ContiguousLayer
import numpy as np
from theano import config, shared, dot, function
import theano.tensor as t
from encoder import AutoEncoder

class ContractiveAutoEncoder(ContiguousLayer, AutoEncoder) :
    '''This class describes a Contractive AutoEncoder (CAE). This is a 
       unsupervised learning technique which encodes an input (feedforward), 
       and then decodes the output vector by sending backwards through the
       layer. This attempts to reconstruct the original input. 

       If the decoded message matches the original input, the encoders is
       considered lossless. Otherwise the loss is calculated and the encoder 
       is updated, so it can better encode the input when encountered again.
       Over time the object will extract regular patterns in the data which
       are frequently encountered.

       CAEs can be used to initialize a Neural Network in a greedy layerwise
       fashion. This should be used to better regularize the weight 
       initialization, and can be used when unlabeled data far outweighs
       the amount of labeled.

       layerID          : unique name identifier for this layer
       input            : the input buffer for this layer
       inputSize        : number of elements in input buffer
       numNeurons       : number of neurons in this layer
       learningRate     : learning rate for all neurons
       contractionRate  : variance (dimensionality) reduction rate
       initialWeights   : weights to initialize the network
                          None generates random weights for the layer
       initialHidThresh : thresholds to initialize the forward network
                          None generates random thresholds for the layer
       initialVisThresh : thresholds to initialize the backward network
                          None generates random thresholds for the layer
       activation       : the sigmoid function to use for activation
                          this must be a function with a derivative form
       randomNumGen     : generator for the initial weight values
    '''
    def __init__ (self, layerID, input, inputSize, numNeurons,
                  learningRate=0.001, contractionRate=0.01,
                  initialWeights=None, initialHidThresh=None,
                  initialVisThresh=None, activation=t.nnet.sigmoid,
                  randomNumGen=None) :
        ContiguousLayer.__init__(self, layerID=layerID,
                                 input=input,
                                 inputSize=inputSize,
                                 numNeurons=numNeurons,
                                 learningRate=learningRate,
                                 initialWeights=initialHidThresh,
                                 activation=activation, 
                                 randomNumGen=randomNumGen)
        AutoEncoder.__init__(contractionRate)
        self._weightsBack = self._weights.T

        # setup initial values for the hidden thresholds
        if initialVisThresh is None :
            initialVisThresh = np.zeros((self.inputSize[1],), 
                                        dtype=config.floatX)
        self._thresholdsBack = shared(value=initialVisThresh, borrow=True)

        # setup the decoder --
        # this take the output of the feedforward process as input and
        # and runs the output back through the network in reverse. The net
        # effect is to reconstruct the input, and ultimately to see how well
        # the network is at encoding the message.
        out = dot(self.output, self._weightsBack) + self._thresholdsBack
        self._decodedInput = out if activation is None else activation(out)
        self._reconstruction = function([self.input], self._decodedInput)

        # compute the jacobian cost of the reconstructed input
        jacobianMat = t.reshape(self.output * (1 - self.output),
                                (self.inputSize[0], 1, self.numNeurons)) * \
                      t.reshape(self._weights, 
                                (1, self.numNeurons, self.inputSize[1]))
        self._jacobianCost = (t.mean(t.sum(jacobianMat ** 2) // 
                             self.inputSize[0])) * self._contractionFactor

        # create the negative log likelihood function --
        # this is our cost function with respect to the original input
        self._nll = \
            t.mean(-t.sum(self.input * t.log(self._decodedInput) +
                          (1 - self.input) * t.log(1 - self._decodedInput), 
                          axis=1))

        gradients = t.grad(self._nll + self._jacobianCost, self.getWeights())
        self._updates = [(weights, weights - learningRate * gradient)
                         for weights, gradient in zip(self.getWeights(), 
                                                      gradients)]

        # TODO: this needs to be stackable and take the input to the first
        #       layer, not just the input of this layer. This will ensure
        #       the other layers are activated to get the input to this layer
        self._trainLayer = t.function([self.input],
                                      [self._jacobianCost, self._nll],
                                      updates=self._updates)

    def getWeights(self) :
        '''Update to account for the decode thresholds.'''
        return [self._weights, self._thresholds, self._thresholdsBack]
    def getUpdates(self) :
        '''This allows the Stacker to build the layerwise training.'''
        return ([self._jacobianCost, self._nll], self._updates)
    def train(self, image) :
        self._trainLayer(image)

