import numpy as np
from theano import config, shared, function
import theano.tensor as t
from ae.encoder import AutoEncoder
from nn.convolutionalLayer import ConvolutionalLayer
from theano.tensor.nnet.conv import conv2d


def max_unpool_2d(input, upsampleFactor) :
    '''Perform perforated upsample from paper : 
       "Image Super-Resolution with Fast Approximate 
        Convolutional Sparse Coding"
    '''
    output_shape = [
        input.shape[1],
        input.shape[2] * upsampleFactor[0],
        input.shape[3] * upsampleFactor[1]
    ]
    stride = input.shape[2]
    offset = input.shape[3]
    in_dim = stride * offset
    out_dim = in_dim * np.prod(upsampleFactor)

    upsamp_matrix = t.zeros((in_dim, out_dim))
    rows = t.arange(in_dim)
    cols = rows * upsampleFactor[0] + \
           (rows / stride * upsampleFactor[1] * offset)
    upsamp_matrix = t.set_subtensor(upsamp_matrix[rows, cols], 1.)

    flat = t.reshape(input, (input.shape[0], output_shape[0], 
                             input.shape[2] * input.shape[3]))
    up_flat = t.dot(flat, upsamp_matrix)
    return t.reshape(up_flat, (input.shape[0], output_shape[0],
                               output_shape[1], output_shape[2]))

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
       initialHidThresh : thresholds to initialize the forward network
                          None generates random thresholds for the layer
       initialVisThresh : thresholds to initialize the backward network
                          None generates random thresholds for the layer
       activation       : the sigmoid function to use for activation
                          this must be a function with a derivative form
       randomNumGen     : generator for the initial weight values
    '''
    def __init__ (self, layerID, input, inputSize, kernelSize, 
                  downsampleFactor, learningRate=0.001, contractionRate=0.01,
                  initialWeights=None, initialHidThresh=None,
                  initialVisThresh=None, activation=t.nnet.sigmoid,
                  randomNumGen=None) :
        ConvolutionalLayer.__init__(self, layerID=layerID,
                                    input=input,
                                    inputSize=inputSize,
                                    kernelSize=kernelSize,
                                    downsampleFactor=downsampleFactor,
                                    learningRate=learningRate,
                                    initialWeights=initialWeights,
                                    initialThresholds=initialHidThresh,
                                    activation=activation, 
                                    randomNumGen=randomNumGen)
        AutoEncoder.__init__(self, contractionRate)
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
        unpooling = max_unpool_2d(self.output, self._downsampleFactor)
        deconvolve = conv2d(unpooling, self._weightsBack,
                            self.getFeatureSize(), kernelBackSize, 
                            border_mode='full')
        out = deconvolve + self._thresholdsBack.dimshuffle('x', 0, 'x', 'x')
        self._decodedInput = out if activation is None else activation(out)
        self._reconstruction = function([self.input], self._decodedInput)

        # create the negative log likelihood function --
        # this is our cost function with respect to the original input
        # NOTE: The jacobian was computed however takes much longer to process
        #       and does not help convergence or regularization. It was removed
        self._nll = t.mean(-t.sum(self.input * t.log(self._decodedInput) +
                           (1 - self.input) * t.log(1 - self._decodedInput), 
                           axis=1))

        gradients = t.grad(self._nll, self.getWeights())
        self._updates = [(weights, weights - learningRate * gradient)
                         for weights, gradient in zip(self.getWeights(), 
                                                      gradients)]

        # TODO: this needs to be stackable and take the input to the first
        #       layer, not just the input of this layer. This will ensure
        #       the other layers are activated to get the input to this layer
        self._trainLayer = function([self.input], self._nll, 
                                    updates=self._updates)

    def getWeights(self) :
        '''Update to account for the decode thresholds.'''
        return [self._weights, self._thresholds, self._thresholdsBack]
    def getUpdates(self) :
        '''This allows the Stacker to build the layerwise training.'''
        return (self._nll, self._updates)

    # DEBUG: For Debugging purposes only 
    def train(self, image) :
        return self._trainLayer(image)
    # DEBUG: For Debugging purposes only 
    def writeWeights(self, ii) :
        import PIL.Image as Image
        from utils import tile_raster_images
        kernelSize = self._weights.get_value(borrow=True).shape
        img = Image.fromarray(tile_raster_images(
        X=np.resize(self._weights.get_value(borrow=True),
                    (kernelSize[0] * kernelSize[1],
                     kernelSize[2], kernelSize[3])),
        img_shape=(kernelSize[2], kernelSize[3]), 
        tile_shape=(kernelSize[0], kernelSize[1]),
        tile_spacing=(1, 1)))
        img.save(self.layerID + '_cae_filters_' + str(ii) + '.png')

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

