from layer import Layer
from profiler import Profiler
import theano.tensor as t
import theano, cPickle, gzip

class ClassifierNetwork () :
    '''The ClassifierNetwork object allows the user to build multi-layer neural
       networks of various topologies easily. This class provides users with 
       functionality to load a trained Network from disk and begin classifying
       inputs. 

       filepath : Path to an already trained network on disk 
                  'None' creates randomized weighting
       log      : Logger to use
    '''
    def __init__ (self, filepath=None, log=None) :
        self._profiler = Profiler(log,
                                  'NeuralNet', 
                                  './NeuralNet-Profile.xml') if \
                                  log is not None else None
        self._layers = []
        self._weights = []
        self._learningRates = []
        self.input = None
        self.output = None
        if filepath is not None :
            self.load(filepath)

    def __getstate__(self) :
        dict = self.__dict__.copy()
        # remove the functions -- they will be rebuilt JIT
        if '_classify' in dict : del dict['_classify']
        # remove the profiler as it is not robust to distributed processing
        dict['_profiler'] = None
        return dict
    def __setstate__(self, dict) :
        # remove any current functions from the object so we force the
        # theano functions to be rebuilt with the new buffers
        if hasattr(self, '_classify') : delattr(self, '_classify')
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
    def load(self, filepath) :
        '''Load the network from disk.
           TODO: This should also support input from Synapse file
        '''
        self._startProfile('Loading network from disk [' + str(filepath) +
                           ']', 'info')
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
           This is used to create several network-wide functions so they will
           be pre-compiled and optimized when we need them. The only function
           across all network types is classify()
        '''
        if len(self._layers) == 0 :
            raise IndexError('Network must have at least one layer' +
                             'to call finalizeNetwork().')

        self._startProfile('Finalizing Network', 'info')

        # create one function that activates the entire network --
        # Here we use softmax on the netowork output to produce a normalized 
        # output prediction, which emphasizes significant neural responses.
        # This takes as its input, the first layer's input, and uses the final
        # layer's output as the function (ie the network classification).
        self._out = t.nnet.softmax(self._layers[-1].output)
        self._classify = theano.function([self._layers[0].input], self._out)

        self._endProfile()
    def classify (self, inputs) :
        '''Classify the given inputs. The input is assumed to be 
           numpy.ndarray with dimensions specified by the first layer of the 
           network.
        '''
        self._startProfile('Classifying the Inputs', 'debug')
        if not hasattr(self, '_classify') :
            self.finalizeNetwork()

        # activating the last layer triggers all previous 
        # layers due to dependencies we've enforced
        out = self._classify(inputs)
        self._endProfile()
        return out

class TrainerNetwork (ClassifierNetwork) :
    '''This network allows for training data on a theano.shared wrapped
       dataset for optimal execution. Because the dataset will be accessed 
       repetitively over the course of training, the shared variables are
       preloaded unto the target architecture. The input to training will be
       an index into this array.

       The network uses a softmax normalization on the output vector to 
       obtain [0,1] classification. This allows for a cross-entropy (ie nll)
       loss function.

       train    : theano.shared dataset used for network training in format --
                  (((numBatches, batchSize, numChannels, rows, cols)), 
                   integerLabelIndices)
       test     : theano.shared dataset used for network testing in format --
                  (((numBatches, batchSize, numChannels, rows, cols)), 
                   integerLabelIndices)
                  The intersection of train and test datasets should be a null
                  set. The test dataset will be used to regularize the training
       regType  : type of regularization term to use
                  default None : perform no additional regularization
                  L1           : Least Absolute Deviation
                  L2           : Least Squares
       filepath : Path to an already trained network on disk 
                  'None' creates randomized weighting
       log      : Logger to use
    '''
    def __init__ (self, train, test, regType='L2', filepath=None, log=None) :
        ClassifierNetwork.__init__(self, filepath=filepath, log=log)
        self._trainData, self._trainLabels = train
        self._testData, self._testLabels = test
        self._regularization = regType
    def __getstate__(self) :
        dict = self.__dict__.copy()
        # remove the functions -- they will be rebuilt JIT
        if '_classify' in dict : del dict['_classify']
        if '_cost' in dict : del dict['_cost']
        if '_trainNetwork' in dict : del dict['_trainNetwork']
        if '_checkAccuracy' in dict : del dict['_checkAccuracy']
        # remove the profiler as it is not robust to distributed processing
        dict['_profiler'] = None
        return dict
    def __setstate__(self, dict) :
        # remove any current functions from the object so we force the
        # theano functions to be rebuilt with the new buffers
        if hasattr(self, '_classify') : delattr(self, '_classify')
        if hasattr(self, '_cost') : delattr(self, '_cost')
        if hasattr(self, '_trainNetwork') : delattr(self, '_trainNetwork')
        if hasattr(self, '_checkAccuracy') : delattr(self, '_checkAccuracy')
        # use the current constructor-supplied profiler --
        # this ensures the profiler is setup for the current system
        tmp = self._profiler
        self.__dict__.update(dict)
        self._profiler = tmp
    def save(self, filepath) :
        '''Save the network to disk.
           TODO: This should also support output to Synapse file
        '''
        self._startProfile('Saving network to disk ['+filepath+']', 'info')
        if '.pkl.gz' in filepath :
            with gzip.open(filepath, 'wb') as f :
                f.write(cPickle.dumps(self.__getstate__(),
                                      protocol=cPickle.HIGHEST_PROTOCOL))
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

        # disable the profiler temporarily so we don't get a second entry
        tmp = self._profiler
        self._profiler = None
        ClassifierNetwork.finalizeNetwork(self)
        self._profiler = tmp

        # create a function to 
        index = t.lscalar('index')
        expectedOutput = t.ivector('expectedOutput')
        numCorrect = t.sum(t.eq(expectedOutput, t.argmax(self._out)))
        self._checkAccuracy = theano.function(
            [index], numCorrect,  
            givens={input: self._testData[index],
                    expectedOutput: self._testLabels[index]})

        # create the negative log likelihood function --
        # This is the cost function for the network, and it assumes [0,1]
        # classification labeling. If the expectedOutput is not [0,1], Doc
        # Brown will hit you with a time machine.
        nll = (-expectedOutput * t.log(self._out) - 
              (1-expectedOutput) * t.log(1-self._out)).mean()
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
            [index], nll, updates=updates,
            givens={input: self._trainData[index],
                    expectedOutput: self._trainLabels[index]})
        self._endProfile()

    def train(self, index) :
        '''Train the network against the pre-loaded inputs. This accepts 
           a batch index into the pre-compiled input and expectedOutput sets.

           NOTE: Class labels for expectedOutput are assumed to be [0,1]
        '''
        self._startProfile('Classifying the Inputs', 'debug')
        if not hasattr(self, '_trainNetwork') :
            self.finalizeNetwork()

        # train the input --
        # the user decides if this is online or batch training
        self._trainNetwork(index)

        self._endProfile()
    def checkAccuracy(self, index) :
        '''Check the accuracy against the pre-compiled the given inputs.
           This accepts a batch index into the pre-compiled input and
           expectedOutput sets.
           This returns the number of correctly classified inputs in the batch.
           The user can find the accuracy by dividing the total number of
           correct by the total number of inputs.
        '''
        self._startProfile('Checking the network Accuracy', 'debug')
        if not hasattr(self, '_checkAccuracy') :
            self.finalizeNetwork()

        # return the sum of all correctly classified targets
        self._checkAccuracy(index)

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
    from datasetUtils import splitToShared

    input = t.fvector('input')
    expectedOutput = t.ivector('expectedOutput')

    numRuns = 10000
    trainArr = ([range(10000)], [[0, 0, 1]])
    testArr  = ([range(10000)], [[0, 0, 1]])
    train = splitToShared(trainArr, borrow=False)
    test  = splitToShared(testArr,  borrow=False)

    network = TrainerNetwork(train=train, test=train, regType='')
    network.addLayer(
        ContiguousLayer('f1', input, len(trainArr[0][0]), 100))
    network.addLayer(
        ContiguousLayer('f2', network.getNetworkOutput(), 100, 3))

    # test the classify runtime
    print "Classifying Inputs..."
    timer = time()
    for i in range(numRuns) :
        out = network.classify(trainArr[0][0])
    timer = time() - timer
    print "total time: " + str(timer) + \
          "s | per input: " + str(timer/numRuns) + "s"
    print (out.argmax(), out)

    # test the train runtime
    numRuns = 10000
    print "Training Network..."
    timer = time()
    for i in range(numRuns) :
        network.train(0)
    timer = time() - timer
    print "total time: " + str(timer) + \
          "s | per input: " + str(timer/numRuns) + "s"
    out = network.classify(trainArr[0][0])
    print (out.argmax(), out)

    network.save('e:/out.pkl.gz')
