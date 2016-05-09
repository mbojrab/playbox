import numpy as np
from theano import config, shared, function
import theano.tensor as t
from ae.encoder import AutoEncoder
from nn.convolutionalLayer import ConvolutionalLayer
from theano.tensor.nnet.conv import conv2d
from nn.costUtils import crossEntropyLoss, meanSquaredLoss, leastSquares

def max_unpool_2d(input, upsampleFactor) :
    '''Perform perforated upsample from paper : 
       "Image Super-Resolution with Fast Approximate 
        Convolutional Sparse Coding"
    '''
    outputShape = [input.shape[1],
                   input.shape[2] * upsampleFactor[0],
                   input.shape[3] * upsampleFactor[1]]
    numElems = input.shape[2] * input.shape[3]
    upsampNumElems = numElems * np.prod(upsampleFactor)

    upsamp = t.zeros((numElems, upsampNumElems))
    rows = t.arange(numElems)
    cols = rows * upsampleFactor[0] + \
           (rows / input.shape[2] * upsampleFactor[1] * input.shape[3])
    upsamp = t.set_subtensor(upsamp[rows, cols], 1.)

    flat = t.reshape(input, (input.shape[0], outputShape[0], 
                             input.shape[2] * input.shape[3]))
    upsampflat = t.dot(flat, upsamp)
    return t.reshape(upsampflat, (input.shape[0], outputShape[0],
                                  outputShape[1], outputShape[2]))

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
       input            : the input buffer for this layer
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
    def __init__ (self, layerID, input, inputSize, kernelSize, 
                  downsampleFactor, learningRate=0.001,
                  dropout=None, contractionRate=None,
                  initialWeights=None, initialHidThresh=None,
                  initialVisThresh=None, activation=t.nnet.sigmoid,
                  randomNumGen=None) :
        ConvolutionalLayer.__init__(self, layerID=layerID,
                                    input=input,
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

        kernelSize = self.getKernelSize()
        kernelBackSize = (kernelSize[1], kernelSize[0], 
                          kernelSize[2], kernelSize[3])
        self._weightsBack = t.reshape(self._weights, 
                                      (kernelBackSize))

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
        unpooling = max_unpool_2d(self.output[1], self._downsampleFactor)
        deconvolve = conv2d(unpooling, self._weightsBack,
                            self.getFeatureSize(), kernelBackSize, 
                            border_mode='full')
        out = deconvolve + self._thresholdsBack.dimshuffle('x', 0, 'x', 'x')
        self._decodedInput = out if activation is None else activation(out)
        self.reconstruction = function([self.input[1]], self._decodedInput)

        # compute the jacobian cost of the output --
        # This works as a sparsity constraint in case the hidden vector is
        # larger than the input vector.
        jacobianMat = conv2d(unpooling * (1 - unpooling), self._weightsBack,
                             self.getFeatureSize(), kernelBackSize, 
                             border_mode='full')
        self._jacobianCost = leastSquares(jacobianMat, self._inputSize[0], 
                                          self._contractionRate)

        # create the negative log likelihood function --
        # this is our cost function with respect to the original input
        # NOTE: The jacobian was computed however takes much longer to process
        #       and does not help convergence or regularization. It was removed
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
            np.resize(self.reconstruction(image), (30, 90)),
            'chip_' + str(ii) + '_reconst.png')
    # DEBUG: For Debugging purposes only 
    def train(self, image) :
        return self._trainLayer(image)


if __name__ == '__main__' :
    import argparse, logging, time
    from nn.datasetUtils import ingestImagery, pickleDataset

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

        import PIL.Image as Image
        from utils import tile_raster_images
        img = Image.fromarray(tile_raster_images(
            X=ae.reconstruction(train[0][0]), img_shape=(28, 28), 
            tile_shape=(10, 10), tile_spacing=(1, 1)))
        img.save('cae_filters_reconstructed_nllOnly_' + str(ii+1) + '.png')

        print 'Epoch [' + str(ii) + ']: ' + str(ae.train(train[0][0])) + \
              ' ' + str(time.time() - start) + 's'

