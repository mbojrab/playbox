import numpy as np
from theano import config, shared, dot, function
import theano.tensor as t
from ae.encoder import AutoEncoder
from nn.contiguousLayer import ContiguousLayer

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
                                 initialWeights=initialWeights,
                                 initialThresholds=initialHidThresh,
                                 activation=activation, 
                                 randomNumGen=randomNumGen)
        AutoEncoder.__init__(self, contractionRate)
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
        out = dot(self.output, self._weightsBack) + self._thresholdsBack
        self._decodedInput = out if activation is None else activation(out)
        self._reconstruction = function([self.input], self._decodedInput)

        # compute the jacobian cost of the reconstructed input
        jacobianMat = t.reshape(self.output * (1 - self.output),
                                (self._inputSize[0], 1, self._numNeurons)) * \
                      t.reshape(self._weights, 
                                (1, self._inputSize[1], self._numNeurons))
        self._jacobianCost = (t.mean(t.sum(jacobianMat ** 2) // 
                             self._inputSize[0])) * self._contractionRate

        # create the negative log likelihood function --
        # this is our cost function with respect to the original input
        self._nll = t.mean(-t.sum(self.input * t.log(self._decodedInput) +
                           (1 - self.input) * t.log(1 - self._decodedInput), 
                           axis=1))

        gradients = t.grad(self._nll + self._jacobianCost, self.getWeights())
        self._updates = [(weights, weights - learningRate * gradient)
                         for weights, gradient in zip(self.getWeights(), 
                                                      gradients)]

        # TODO: this needs to be stackable and take the input to the first
        #       layer, not just the input of this layer. This will ensure
        #       the other layers are activated to get the input to this layer
        self._trainLayer = function([self.input], 
                                    [self._jacobianCost, self._nll],
                                    updates=self._updates)

    def getWeights(self) :
        '''Update to account for the decode thresholds.'''
        return [self._weights, self._thresholds, self._thresholdsBack]
    def getUpdates(self) :
        '''This allows the Stacker to build the layerwise training.'''
        return ([self._jacobianCost, self._nll], self._updates)

    # DEBUG: For Debugging purposes only 
    def train(self, image) :
        return self._trainLayer(image)
    # DEBUG: For Debugging purposes only 
    def writeWeights(self) :
        import PIL.Image as Image
        from utils import tile_raster_images
        img = Image.fromarray(tile_raster_images(
        X=self._weights.get_value(borrow=True).T,
        img_shape=(28, 28), tile_shape=(10, 10),
        tile_spacing=(1, 1)))
        img.save('cae_filters.png')

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
    ae = ContractiveAutoEncoder('cae', input, 
                                (train[0].shape[1], train[0].shape[2]),
                                options.neuron, options.learn,
                                options.contraction)
    for ii in range(10) :
        start = time.time()
        for jj in range(2) :
            ae.train(train[0][jj])
        print 'Epoch [' + str(ii) + ']: ' + str(ae.train(train[0][0])) + \
              ' ' + str(time.time() - start) + 's'