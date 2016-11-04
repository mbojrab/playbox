import numpy as np
from theano import config, shared, function
import theano.tensor as t
from ae.encoder import AutoEncoder
from nn.convolutionalLayer import ConvolutionalLayer
from theano.tensor.nnet.conv import conv2d

class ConvolutionalAutoEncoder(ConvolutionalLayer, AutoEncoder) :
    '''This class describes a Contractive AutoEncoder (CAE) for a convolutional
       layer. This differs from the normal CAE 

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
       downsampleFactor : (rowFactor, columnFactor)
       learningRate     : learning rate for all neurons
       contractionRate  : variance (dimensionality) reduction rate
                          None uses '1 / numNeurons'
       dropout          : rate of retention in a given neuron during training
                          NOTE: input layers should be around .8 or .9
                                hidden layers should be around .5 or .6
                                output layers should always be 1.
       initialHidThresh : thresholds to initialize the forward network
                          None generates random thresholds for the layer
       initialVisThresh : thresholds to initialize the backward network
                          None generates random thresholds for the layer
       activation       : the sigmoid function to use for activation
                          this must be a function with a derivative form
       randomNumGen     : generator for the initial weight values
    '''
    def __init__ (self, layerID, inputSize, kernelSize, 
                  downsampleFactor, learningRate=0.001,
                  dropout=None, contractionRate=None,
                  initialWeights=None, initialHidThresh=None,
                  initialVisThresh=None, activation=t.nnet.sigmoid,
                  randomNumGen=None) :
        ConvolutionalLayer.__init__(self, layerID=layerID,
                                    inputSize=inputSize,
                                    kernelSize=kernelSize,
                                    downsampleFactor=downsampleFactor,
                                    learningRate=learningRate,
                                    dropout=dropout,
                                    initialWeights=initialWeights,
                                    initialThresholds=initialHidThresh,
                                    activation=activation, 
                                    randomNumGen=randomNumGen)
        AutoEncoder.__init__(self, 1. / np.prod(kernelSize[:]) if \
                                   contractionRate is None else \
                                   contractionRate)

        # setup initial values for the hidden thresholds
        if initialVisThresh is None :
            initialVisThresh = np.zeros((self._inputSize[1],), 
                                        dtype=config.floatX)
        self._thresholdsBack = shared(value=initialVisThresh, borrow=True)

    def _setActivation(self, out) :
        from theano.tensor import round
        return round(out) if self._activation is None else \
               round(self._activation(out))

    def __getstate__(self) :
        '''Save network pickle'''
        from dataset.shared import fromShared
        dict = ConvolutionalLayer.__getstate__(self)
        dict['_thresholdsBack'] = fromShared(self._thresholdsBack)
        # remove the functions -- they will be rebuilt JIT
        if 'reconstruction' in dict : del dict['reconstruction']
        if '_costs' in dict : del dict['_costs']
        if '_updates' in dict : del dict['_updates']
        if 'trainLayer' in dict : del dict['trainLayer']
        return dict

    def __setstate__(self, dict) :
        '''Load network pickle'''
        from theano import shared
        # remove any current functions from the object so we force the
        # theano functions to be rebuilt with the new buffers
        if hasattr(self, 'reconstruction') : delattr(self, 'reconstruction')
        if hasattr(self, '_costs') : delattr(self, '_costs')
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
        op = t.signal.pool.Pool(ds=upsampleFactor, ignore_border=True, 
                                st=None, padding=(0, 0), mode='max')
        return op.grad((self._prePoolingInput[1],), (input,))[0]

    def _getWeightsBack(self) :
        '''Calculate the weights used for decoding.'''
        kernelSize = self.getKernelSize()
        kernelBackSize = (kernelSize[1], kernelSize[0], 
                          kernelSize[2], kernelSize[3])
        return t.reshape(self._weights, (kernelBackSize))

    def _decode(self, input) :
        weightsBack = self._getWeightsBack()
        deconvolve = conv2d(input, weightsBack, self.getFeatureSize(), 
                            weightsBack.shape.eval(), border_mode='full')
        out = deconvolve + self._thresholdsBack.dimshuffle('x', 0, 'x', 'x')
        return out if self._activation is None else self._activation(out)

    def finalize(self, networkInput, layerInput) :
        '''Setup the computation graph for this layer.
           networkInput : the input variable tuple for the network
                          format (inClass, inTrain)
           layerInput   : the input variable tuple for this layer
                          format (inClass, inTrain)
        '''
        from nn.costUtils import calcLoss, leastSquares, calcSparsityConstraint
        ConvolutionalLayer.finalize(self, networkInput, layerInput)

        weightsBack = self._getWeightsBack()

        # setup the decoder --
        # this take the output of the feedforward process as input and
        # and runs the output back through the network in reverse. The net
        # effect is to reconstruct the input, and ultimately to see how well
        # the network is at encoding the message.
        unpooling = self._unpool_2d(self.output[1], self._downsampleFactor)
        decodedInput = self._decode(unpooling)

        # DEBUG: For Debugging purposes only
        self.reconstruction = function([networkInput[0]], decodedInput)

        sparseConstr = calcSparsityConstraint(self.output[0], 
                                              self.getOutputSize())

        # compute the jacobian cost of the output --
        # This works as a sparsity constraint in case the hidden vector is
        # larger than the input vector.
        jacobianMat = conv2d(unpooling * (1 - unpooling), weightsBack,
                             self.getFeatureSize(), weightsBack.shape.eval(), 
                             border_mode='full')
        jacobianCost = leastSquares(jacobianMat, self._inputSize[0], 
                                    self._contractionRate)

        # create the negative log likelihood function --
        # this is our cost function with respect to the original input
        # NOTE: The jacobian was computed however takes much longer to process
        #       and does not help convergence or regularization. It was removed
        cost = calcLoss(self.input[0], decodedInput, self._activation) / \
               self.getInputSize()[0]
        self._costs = [cost, jacobianCost, sparseConstr]

        gradients = t.grad(t.sum(self._costs), self.getWeights())
        self._updates = [(weights, weights - self._learningRate * gradient)
                         for weights, gradient in zip(self.getWeights(), 
                                                      gradients)]

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
        return self._decode(self._unpool_2d(
            t.reshape(input, self.getOutputSize()), self._downsampleFactor))

    def getWeights(self) :
        '''Update to account for the decode thresholds.'''
        return [self._weights, self._thresholds, self._thresholdsBack]
    def getUpdates(self) :
        '''This allows the Stacker to build the layerwise training.'''
        return (self._costs, self._updates)

    # DEBUG: For Debugging purposes only
    def saveReconstruction(self, image, ii) :
        from dataset.debugger import saveNormalizedImage
        saveNormalizedImage(np.resize(self.reconstruction(image), (28, 28)),
                            'chip_' + str(ii) + '_reconst.png')


if __name__ == '__main__' :
    import argparse, logging, time
    from dataset.reader import ingestImagery, pickleDataset
    from dataset.debugger import saveTiledImage

    parser = argparse.ArgumentParser()
    parser.add_argument('--log', dest='logfile', type=str, default=None,
                        help='Specify log output file.')
    parser.add_argument('--level', dest='level', default='INFO', type=str, 
                        help='Log Level.')
    parser.add_argument('--contraction', dest='contraction', default=0.1, 
                        type=float, help='Rate of contraction.')
    parser.add_argument('--learn', dest='learn', type=float, default=0.01,
                        help='Rate of learning on AutoEncoder.')
    parser.add_argument('--kernel', dest='kernel', type=int, default=6,
                        help='Number of kernels in Convolutional Layer.')
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

    input = t.ftensor4()
    ae = ConvolutionalAutoEncoder('cae', input, train[0].shape[1:], 
                                  (options.kernel,train[0].shape[2],5,5), 
                                  (2,2))
    ae.writeWeights(0)
    for ii in range(100) :
        start = time.time()
        for jj in range(len(train[0])) :
            ae.train(train[0][jj])
        ae.writeWeights(ii+1)

        saveTiledImage(
            image=ae.reconstruction(train[0][0]),
            path=ae.layerID + '_cae_filters_reconstructed_' +
                 str(ii+1) + '.png', imageShape=(28, 28), spacing=1,
            interleave=True)
