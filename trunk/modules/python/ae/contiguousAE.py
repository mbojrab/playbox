import numpy as np
from theano import config, shared, dot, function
import theano.tensor as t
from ae.encoder import AutoEncoder
from nn.contiguousLayer import ContiguousLayer
from nn.costUtils import crossEntropyLoss, meanSquaredLoss
from nn.costUtils import computeJacobian, leastSquares

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

       CAEs can be stacked and trained in a greedy layerwise manner, and the
       trained CAEs can be used to initialize a Neural Network into a better
       regularized state than random initialization. Lastly this technique can
       be used when the amount of unlabeled data far outweighs the amount of 
       labeled data.

       layerID          : unique name identifier for this layer
       input            : the input buffer for this layer
       inputSize        : number of elements in input buffer
       numNeurons       : number of neurons in this layer
       learningRate     : learning rate for all neurons
       contractionRate  : variance (dimensionality) reduction rate
                          None uses '1 / numNeurons'
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
                  learningRate=0.001, dropout=None, contractionRate=None,
                  initialWeights=None, initialHidThresh=None,
                  initialVisThresh=None, activation=t.nnet.sigmoid,
                  randomNumGen=None) :
        ContiguousLayer.__init__(self, layerID=layerID,
                                 input=input,
                                 inputSize=inputSize,
                                 numNeurons=numNeurons,
                                 learningRate=learningRate,
                                 dropout=dropout,
                                 initialWeights=initialWeights,
                                 initialThresholds=initialHidThresh,
                                 activation=activation, 
                                 randomNumGen=randomNumGen)
        AutoEncoder.__init__(self, 1. / numNeurons if contractionRate is None \
                                   else contractionRate)

        # use tied weights for the weight matrix
        self._weightsBack = self._weights.T

        # setup initial values for the hidden thresholds
        if initialVisThresh is None :
            initialVisThresh = np.zeros((self._inputSize[1],), 
                                        dtype=config.floatX)
        self._thresholdsBack = shared(value=initialVisThresh, borrow=True)

        # setup the decoder --
        # this take the output of the feedforward process as input and
        # and runs the output back through the network in reverse. The net
        # effect is to reconstruct the input, and ultimately to see how well
        # the network is at encoding the message.
        out = dot(self.output[1], self._weightsBack) + self._thresholdsBack
        self._decodedInput = out if activation is None else activation(out)
        self.reconstruction = function([self.input[1]], self._decodedInput)

        # compute the jacobian cost of the output --
        # This works as a sparsity constraint in case the hidden vector is
        # larger than the input vector.
        self._jacobianCost = \
        leastSquares(computeJacobian(self.output[1], self._weights,
                                     self._inputSize[0], self._inputSize[1],
                                     self._numNeurons), 
                     self._inputSize[0], self._contractionRate)

        # create the negative log likelihood function --
        # this is our cost function with respect to the original input
        if activation == t.nnet.sigmoid :
            self._cost = crossEntropyLoss(self.input[1], self._decodedInput, 1)
        else :
            self._cost = meanSquaredLoss(self.input[1], self._decodedInput)

        gradients = t.grad(self._cost + self._jacobianCost, self.getWeights())
        self._updates = [(weights, weights - learningRate * gradient)
                         for weights, gradient in zip(self.getWeights(), 
                                                      gradients)]

        # TODO: this needs to be stackable and take the input to the first
        #       layer, not just the input of this layer. This will ensure
        #       the other layers are activated to get the input to this layer
        self._trainLayer = function([self.input[1]], 
                                    [self._cost, self._jacobianCost],
                                    updates=self._updates)

    def getWeights(self) :
        '''Update to account for the decode thresholds.'''
        return [self._weights, self._thresholds, self._thresholdsBack]
    def getUpdates(self) :
        '''This allows the Stacker to build the layerwise training.'''
        return ([self._cost, self._jacobianCost], self._updates)

    # DEBUG: For Debugging purposes only
    def saveReconstruction(self, image, ii) :
        import ae.utils
        ae.utils.saveNormalizedImage(
            np.resize(self.reconstruction(image), (30, 30)),
            'chip_' + str(ii) + '_reconst.png')
    # DEBUG: For Debugging purposes only
    def train(self, image) :
        return self._trainLayer(image)

if __name__ == '__main__' :
    import argparse, logging, time
    from nn.datasetUtils import ingestImagery, pickleDataset
    from nn.debugger import saveTiledImage

    parser = argparse.ArgumentParser()
    parser.add_argument('--log', dest='logfile', type=str, default=None,
                        help='Specify log output file.')
    parser.add_argument('--level', dest='level', default='INFO', type=str, 
                        help='Log Level.')
    parser.add_argument('--contraction', dest='contraction', default=0.1, 
                        type=float, help='Rate of contraction.')
    parser.add_argument('--learn', dest='learn', type=float, default=0.01,
                        help='Rate of learning on AutoEncoder.')
    parser.add_argument('--dropout', dest='dropout', type=bool, default=False,
                        help='Perform dropout on the layer.')
    parser.add_argument('--neuron', dest='neuron', type=int, default=500,
                        help='Number of Neurons in Hidden Layer.')
    parser.add_argument('data', help='Directory or pkl.gz file for the ' +
                                     'training and test sets')
    options = parser.parse_args()

    # setup the logger
    log = logging.getLogger('CAE: ' + options.data)
    log.setLevel(options.level.upper())
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    stream = logging.StreamHandler()
    stream.setLevel(options.level.upper())
    stream.setFormatter(formatter)
    log.addHandler(stream)

    # NOTE: The pickleDataset will silently use previously created pickles if
    #       one exists (for efficiency). So watch out for stale pickles!
    train, test, labels = ingestImagery(pickleDataset(
            options.data, batchSize=100, 
            holdoutPercentage=0, log=log), shared=False, log=log)
    vectorized = (train[0].shape[0], train[0].shape[1], 
                  train[0].shape[3] * train[0].shape[4])
    train = (np.reshape(train[0], vectorized), train[1])

    input = t.fmatrix()
    ae = ContractiveAutoEncoder('cae', input=input, 
                                inputSize=(train[0].shape[1],
                                           train[0].shape[2]),
                                numNeurons=options.neuron,
                                learningRate=options.learn,
                                dropout=.5 if options.dropout else 1.)
    for ii in range(50) :
        start = time.time()
        for jj in range(len(train[0])) :
            ae.train(train[0][jj])

        ae.writeWeights(ii+1, (28,28))

        saveTiledImage(image=ae.reconstruction(train[0][0]),
                       path='cae_filters_reconstructed_' + str(ii+1) + '.png',
                       imageShape=(28, 28), spacing=1)
        print 'Epoch [' + str(ii) + ']: ' + str(ae.train(train[0][0])) + \
              ' ' + str(time.time() - start) + 's'