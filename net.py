from layer import Layer
from profiler import Profiler
import theano.tensor as t
import theano, cPickle, gzip

class Net () :
    '''The Net object allows the user to build multi-layer neural networks of
       various topologies easily. The Net object creates optimized functions
       for network-wide classification and back propagation. The network uses
       a softmax normalization on the output vector to obtain [0,1]
       classification. This allows for a cross-entropy (ie nll) loss function.

       regType : type of regularization term to use
                 default None : perform no additional regularization
                 L1           : Least Absolute Deviation
                 L2           : Least Squares
       log     : Logger to use
    '''
    def __init__ (self, regType='L2', log=None) :
        self._profiler = Profiler(log,
                                  'NeuralNet', 
                                  './NeuralNet-Profile.xml') if \
                                  log is not None else None
        self._regularization = regType
        self._layers = []
        self._weights = []
        self._learningRates = []
        self.input = None
        self.output = None

    def __getstate__(self) :
        dict = self.__dict__.copy()
        # remove the functions -- they will be rebuilt JIT
        if '_classify' in dict :
            del dict['_classify']
        if '_cost' in dict :
            del dict['_cost']
        if '_trainNetwork' in dict :
            del dict['_trainNetwork']
        # remove the profiler as it is not robust to distributed processing
        dict['_profiler'] = None
        return dict
    def __setstate__(self, dict) :
        # remove any current functions from the object so we force the
        # theano functions to be rebuilt with the new buffers
        if hasattr(self, '_classify') :
            delattr(self, '_classify')
        if hasattr(self, '_cost') :
            delattr(self, '_cost')
        if hasattr(self, '_trainNetwork') :
            delattr(self, '_trainNetwork')
        # use the current constructor-supplied profiler --
        # this ensures the profiler is setup for the current system
        tmp = self._profiler
        self.__dict__.update(dict)
        self._profiler = tmp
    def _startProfile(self, message, level) :
        if self._profiler is not None :
            self._profiler.startProfile(message, level)
    def _endProfile(self) :
        if self._profiler is not None : 
            self._profiler.endProfile()
    def _listify(self, data) :
        if data is None : return []
        else : return data if isinstance(data, list) else [data]

    def getNetworkInput(self) :
        '''Return the first layer's input'''
        if len(self._layers) == 0 :
            raise IndexError('Network must have at least one layer' +
                             'to call getNetworkInput().')
        return self._layers[0].input
    def getNetworkInputSize(self) :
        '''Return the first layer's input size'''
        if len(self._layers) == 0 :
            raise IndexError('Network must have at least one layer' +
                             'to call getNetworkInputSize().')
        return self._layers[0].getInputSize()
    def getNetworkOutput(self) :
        '''Return the last layer's output. This should be used as input to
           the next layer.
        '''
        if len(self._layers) == 0 :
            raise IndexError('Network must have at least one layer' +
                             'to call getNetworkOutput().')
        return self._layers[-1].output
    def getNetworkOutputSize(self) :
        '''Return the last layer's output size.'''
        if len(self._layers) == 0 :
            raise IndexError('Network must have at least one layer' +
                             'to call getNetworkOutputSize().')
        return self._layers[-1].getOutputSize()
    def save(self, filepath) :
        '''Save the network to disk.
           TODO: This should instead pickle only the weights and state
                 Performing the Pickling in this manner could have issues
                 with future releases of Theano.
           TODO: This should also support output to Synapse file
        '''
        self._startProfile('Saving network to disk ['+filepath+']', 'info')
        if '.pkl.gz' in filepath :
            with gzip.open(filepath, 'wb') as f :
                f.write(cPickle.dumps(self.__getstate__(),
                                      protocol=cPickle.HIGHEST_PROTOCOL))
        self._endProfile()
    def load(self, filepath) :
        '''Load the network from disk.
           TODO: This should instead pickle the weights and build layers
                 Performing the Pickling in this manner could have issues
                 with future releases of Theano.
           TODO: This should also support input from Synapse file
        '''
        self._startProfile('Loading network from disk ['+filepath+']', 'info')
        if '.pkl.gz' in filepath :
            with gzip.open(filepath, 'rb') as f :
                self.__setstate__(cPickle.load(f))
        self._endProfile()
    def addLayer(self, layer) :
        '''Add a Layer to the network. It is the responsibility of the user
           to connect the current network's output as the input to the next
           layer.
        '''
        if not isinstance(layer, Layer) :
            raise TypeError('addLayer is expecting a Layer object.')
        self._startProfile('Adding a layer to the network', 'debug')

        # add it to our layer list
        self._layers.append(layer)
        
        # weight/bias are added in reverse order because they will
        # be used back propagation, which runs output to input
        self._weights = layer.getWeights() + self._weights
        self._learningRates = [layer.getLearningRate()] * 2 + \
                              self._learningRates
        self._endProfile()

    def finalizeNetwork(self) :
        '''Setup the network based on the current network configuration.
           This creates several network-wide functions so they will be
           pre-compiled and optimized when we need them.
        '''
        if len(self._layers) == 0 :
            raise IndexError('Network must have at least one layer' +
                             'to call getNetworkInput().')

        self._startProfile('Finalizing Network', 'info')

        # create one function that activates the entire network --
        # Here we use softmax on the netowork output to produce a normalized 
        # output prediction, which emphasizes significant neural responses.
        # This takes as its input, the first layer's input, and uses the final
        # layer's output as the function (ie the network classification).
        out = t.nnet.softmax(self._layers[-1].output)
        self._classify = theano.function([self._layers[0].input], out)

        # create the negative log likelihood function --
        # This is the cost function for the network, and it assumes [0,1]
        # classification labeling. If the expectedOutput is not [0,1], Doc
        # Brown will hit you with a time machine.
        expectedOutput = t.ivector('expectedOutput')
        nll = (-expectedOutput * t.log(out) - 
              (1-expectedOutput) * t.log(1-out)).mean()
        self._cost = theano.function([self._layers[0].input, expectedOutput],
                                     nll)

        # calculate a regularization term -- if desired
        reg = 0.0
        regSF = 0.0001
        # L1-norm provides 'Least Absolute Deviation' --
        # built for sparse outputs and is resistent to outliers
        if self._regularization == 'L1' :
            reg = sum([abs(w).sum() for w in self._weights]) * regSF
        # L2-norm provides 'Least Squares' --
        # built for dense outputs and is computationally stable at small errors
        elif self._regularization == 'L2' :
            reg = sum([abs(w ** 2).sum() for w in self._weights]) * regSF


        # create the function for back propagation of all layers --
        # this is combined for convenience
        gradients = t.grad(nll + reg, self._weights)
        updates = [(weights, weights - learningRate * gradient)
                   for weights, gradient, learningRate in \
                       zip(self._weights, gradients, self._learningRates)]
        self._trainNetwork = theano.function(
            [self._layers[0].input, expectedOutput], nll, updates=updates)
        self._endProfile()

    def classify (self, input) :
        '''Classify the given input. This lets the user classify the output.'''
        self._startProfile('Classifying the Input', 'debug')
        if not hasattr(self, '_classify') :
            self.finalizeNetwork()

        # activating the last layer triggers all previous 
        # layers due to dependencies we've enforced
        out = self._classify(input)
        self._endProfile()
        return out

    def train(self, inputs, expectedOutputs) :
        '''Train the network against this inputs. This accepts single input or
           full lists of inputs. Depending on what is sent in (and how the
           network is initialized) this is online or batch training.

           NOTE: Class labels for expectedOutput are assumed to be [0,1]
        '''
        self._startProfile('Classifying the Inputs', 'debug')
        if not hasattr(self, '_trainNetwork') :
            self.finalizeNetwork()

        # train the input --
        # the user decides if this is online or batch training
        self._trainNetwork(inputs, expectedOutputs)

        self._endProfile()


if __name__ == "__main__" :
    '''
    # this shows the execution of a sequence of functions by
    # combining them into a higher level function
    a = t.iscalar('a')
    b = t.iscalar('b')

    out1 = a**2
    square = theano.function([a], out1)

    out2 = out1**3
    cube = theano.function([out1], out2)

    total = theano.function([a], out2)

    print square(4)
    print cube(4)
    print total(4)
    '''

    from contiguousLayer import ContiguousLayer
    from time import time

    input = t.fvector('input')
    expectedOutput = t.bvector('expectedOutput')

    network = Net(regType='')
    network.addLayer(
        ContiguousLayer('c1', input, 5, 3))
    network.addLayer(
        ContiguousLayer('c2', network.getNetworkOutput(), 3, 3))

    numRuns = 10000
    arr = [1, 2, 3, 4, 5]
    exp = [0, 0, 1]

    # test the classify runtime
    print "Classifying Inputs..."
    timer = time()
    for i in range(numRuns) :
        out = network.classify(arr)
    timer = time() - timer
    print "total time: " + str(timer) + \
          "s | per input: " + str(timer/numRuns) + "s"
    print (out.argmax(), out)

    # test the train runtime
#    numRuns = 10000
    print "Training Network..."
    timer = time()
    for i in range(numRuns) :
        network.train(arr, exp)
    timer = time() - timer
    print "total time: " + str(timer) + \
          "s | per input: " + str(timer/numRuns) + "s"
    out = network.classify(arr)
    print (out.argmax(), out)

    network.save('e:/out.pkl.gz')
