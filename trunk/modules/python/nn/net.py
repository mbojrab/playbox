from nn.layer import Layer
import theano.tensor as t
import theano
from dataset.pickle import writePickleZip, readPickleZip
from dataset.shared import isShared, getShape

class Network () :
    def __init__ (self, prof=None, debug=False) :
        self._profiler = prof
        self._layers = []
        self._debug = debug

    def __getstate__(self) :
        '''Save network pickle'''
        dict = self.__dict__.copy()
        # remove the profiler as it is not robust to distributed processing
        dict['_profiler'] = None
        dict['_debug'] = None
        return dict

    def __setstate__(self, dict) :
        '''Load network pickle'''
        # use the current constructor-supplied parameters --
        # this ensures the network is setup for the current system
        tmpProf = self._profiler
        tmpDebug = self._debug
        self.__dict__.update(dict)
        self._profiler = tmpProf
        self._debug = tmpDebug

    def _startProfile(self, message, level) :
        '''Start a profile if the profiler exists.'''
        if self._profiler is not None :
            self._profiler.startProfile(message, level)

    def _endProfile(self) :
        '''End the last profile.'''
        if self._profiler is not None :
            self._profiler.endProfile()

    def _listify(self, data) :
        if data is None : return []
        else : return data if isinstance(data, list) else [data]

    def finalizeNetwork(self, networkInput) :
        '''Setup the network based on the current network configuration.'''
        if len(self._layers) == 0 :
            raise IndexError('Network must have at least one layer' +
                             'to call finalizeNetwork().')

        self._startProfile('Finalizing Network', 'info')

        # finalize the layers to create the computational graphs
        networkInput = (networkInput, networkInput) \
                     if not isinstance(networkInput, tuple) else networkInput
        layerInput = networkInput
        for layer in self._layers :
            self._startProfile('Finalizing Layer [' + layer.layerID + ']',
                               'debug')
            layer.finalize(networkInput, layerInput)
            layerInput = layer.output
            self._endProfile()
        self._endProfile()

    def save(self, filepath) :
        '''Save the network to disk.
           TODO: This should also support output to Synapse file
        '''
        self._startProfile('Saving network to disk [' + filepath + ']', 'info')
        if '.pkl.gz' in filepath :
            writePickleZip(filepath, self.__getstate__())
        self._endProfile()

    def load(self, filepath) :
        '''Load the network from disk.
           TODO: This should also support input from Synapse file
        '''
        self._startProfile('Loading network from disk [' + str(filepath) +
                           ']', 'info')
        self.__setstate__(readPickleZip(filepath))
        self._endProfile()

    def addLayer(self, layer) :
        '''Add a Layer to the network.'''
        if not isinstance(layer, Layer) :
            raise TypeError('addLayer is expecting a Layer object.')
        self._startProfile('Adding a layer to the network', 'debug')
        self._layers.append(layer)
        self._endProfile()

    def removeLayer(self) :
        '''Remove the last layer from the network.
           NOTE: This purposefully does not allow a layer index.
           NOTE: Use this method for Transfer Learning or SAE network tuning.
        '''
        self._startProfile('Removing the last layer from the network', 'debug')
        del self._layers[-1]
        self._endProfile()

    def getNumLayers(self) :
        '''Return the total number of layers in the network.'''
        return len(self._layers)

    def getNetworkInput(self) :
        '''Return the first layer's input'''
        if len(self._layers) == 0 :
            raise IndexError('Network must have at least one layer ' +
                             'to call getNetworkInput().')
        return self._layers[0].input

    def getNetworkInputSize(self) :
        '''Return the first layer's input size'''
        if len(self._layers) == 0 :
            raise IndexError('Network must have at least one layer ' +
                             'to call getNetworkInputSize().')
        return self._layers[0].getInputSize()

    def getNetworkOutput(self) :
        '''Return the last layer's output. This should be used as input to
           the next layer.
        '''
        if len(self._layers) == 0 :
            raise IndexError('Network must have at least one layer ' +
                             'to call getNetworkOutput().')
        return self._layers[-1].output

    def getNetworkOutputSize(self) :
        '''Return the last layer's output size.'''
        if len(self._layers) == 0 :
            raise IndexError('Network must have at least one layer ' +
                             'to call getNetworkOutputSize().')
        return self._layers[-1].getOutputSize()


class ClassifierNetwork (Network) :
    '''The ClassifierNetwork object allows the user to build multi-layer neural
       networks of various topologies easily. This class provides users with
       functionality to load a trained Network from disk and begin classifying
       inputs.

       filepath    : Path to an already trained network on disk
                     'None' creates randomized weighting
       prof        : Profiler to use
       debug       : Turn on debugging information
    '''
    def __init__ (self, filepath=None, prof=None, debug=False) :
        Network.__init__(self, prof, debug)

        # NOTE: this must be the last thing performed in init
        if filepath is not None :
            self.load(filepath)

    def __getstate__(self) :
        '''Save network pickle'''
        dict = Network.__getstate__(self)
        # remove the functions -- they will be rebuilt JIT
        if '_outClassMax' in dict : del dict['_outClassMax']
        if '_classify' in dict : del dict['_classify']
        if '_classifyAndSoftmax' in dict : del dict['_classifyAndSoftmax']
        return dict

    def __setstate__(self, dict) :
        '''Load network pickle'''
        # remove any current functions from the object so we force the
        # theano functions to be rebuilt with the new buffers
        if hasattr(self, '_outClassMax') :
            delattr(self, '_outClassMax')
        if hasattr(self, '_classify') :
            delattr(self, '_classify')
        if hasattr(self, '_classifyAndSoftmax') :
            delattr(self, '_classifyAndSoftmax')
        Network.__setstate__(self, dict)

    def finalizeNetwork(self, networkInput) :
        '''Setup the network based on the current network configuration.
           This is used to create several network-wide functions so they will
           be pre-compiled and optimized when we need them. The only function
           across all network types is classify()
        '''
        self._startProfile('Finalizing Network', 'info')

        # disable the profiler temporarily so we don't get a second entry
        tmp = self._profiler
        self._profiler = None
        Network.finalizeNetwork(self, networkInput)
        self._profiler = tmp

        # create one function that activates the entire network --
        # Here we use softmax on the network output to produce a normalized
        # output prediction, which emphasizes significant neural responses.
        # This takes as its input, the first layer's input, and uses the final
        # layer's output as the function (ie the network classification).
        outClass = t.nnet.softmax(self.getNetworkOutput()[0])
        self._outClassMax = t.argmax(outClass, axis=1)
        self._classify = theano.function([self.getNetworkInput()[0]],
                                         self._outClassMax)
        self._classifyAndSoftmax = theano.function(
            [self.getNetworkInput()[0]],
            [self._outClassMax, outClass])
        self._endProfile()

    def classify (self, inputs) :
        '''Classify the given inputs. The input is assumed to be
           numpy.ndarray with dimensions specified by the first layer of the
           network. The output is the index of the softmax classification.
        '''
        self._startProfile('Classifying the Inputs', 'debug')
        if not hasattr(self, '_classify') :
            from dataset.shared import toShared
            inp = toShared(inputs, borrow=True) \
                  if not isShared(inputs) else inputs
            self.finalizeNetwork(inp[:])

        # activating the last layer triggers all previous
        # layers due to dependencies we've enforced
        classIndex = self._classify(inputs)
        self._endProfile()
        return classIndex

    def classifyAndSoftmax (self, inputs) :
        '''Classify the given inputs. The input is assumed to be
           numpy.ndarray with dimensions specified by the first layer of the
           network.

           return : (classification index, softmax vector)
        '''
        self._startProfile('Classifying the Inputs', 'debug')
        if not hasattr(self, '_classifyAndSoftmax') :
            from dataset.shared import toShared
            inp = toShared(inputs, borrow=True) \
                  if not isShared(inputs) else inputs
            self.finalizeNetwork(inp[:])

        # activating the last layer triggers all previous
        # layers due to dependencies we've enforced
        classIndex, softmax = self._classifyAndSoftmax(inputs)
        self._endProfile()
        return classIndex, softmax


class LabeledClassifierNetwork (ClassifierNetwork) :
    '''The LabeledClassifierNetwork adds labeling to the classification.

       labels      : Labels for the classification layer
       filepath    : Path to an already trained network on disk
                     'None' creates randomized weighting
       prof        : Profiler to use
       debug       : Turn on debugging information
    '''
    def __init__ (self, labels=None, filepath=None, prof=None, debug=False) :
        ClassifierNetwork.__init__(self, filepath, prof, debug)

        # if we loaded a synapse use the labels from the pickle
        if filepath is None :
            # convert the labels to a simple list. This allows various data
            # types to be supported here. A list is the easiest to save.
            self._networkLabels = labels if isinstance(labels, list) else \
                                  list(labels)

    def getNetworkLabels(self) :
        '''Return the Labels for the network. All other interactions with
           training and accuracy deal with the label index, so this decodes
           it into a string classification.
        '''
        return self._networkLabels

    def convertToLabels(self, labelIndices) :
        '''Return the string labels for a vector of indices.'''
        return [self._networkLabels[ii] for ii in labelIndices]


class TrainerNetwork (LabeledClassifierNetwork) :
    '''This network allows for training data on a theano.shared wrapped
       dataset for optimal execution. Because the dataset will be accessed
       repetitively over the course of training, the shared variables are
       preloaded unto the target architecture. The input to training will be
       an index into this array.

       The network uses a softmax normalization on the output vector to
       obtain [0,1] classification. This allows for a cross-entropy (ie nll)
       loss function.

       train    : theano.shared dataset used for network training in format
                  NOTE: Currently the user is allowed to pass two formats
                        for this field. --

                        (((numBatches, batchSize, numChannels, rows, cols)),
                          (numBatches, oneHotIndex))

                        (((numBatches, batchSize, numChannels, rows, cols)),
                          (numBatches, batchSize, expectedOutputVect))

       test     : theano.shared dataset used for network testing in format --
                  (((numBatches, batchSize, numChannels, rows, cols)),
                  integerLabelIndices)
                  The intersection of train and test datasets should be a null
                  set. The test dataset will be used to regularize the training
       regType  : type of regularization term to use
                  default None : perform no additional regularization
                  L1           : Least Absolute Deviation
                  L2           : Least Squares
       regSF    : regularization scale factor
                  NOTE: a good value is 1. / numTotalNeurons
       filepath : Path to an already trained network on disk
                  'None' creates randomized weighting
       prof     : Profiler to use
       debug    : Turn on debugging information
    '''
    def __init__ (self, train, test, labels, regType='L2', regScaleFactor=0.,
                  filepath=None, prof=None, debug=False) :
        from nn.reg import Regularization
        LabeledClassifierNetwork.__init__(self, labels, filepath=filepath,
                                          prof=prof, debug=debug)
        self._trainData, self._trainLabels = train
        self._testData, self._testLabels = test

        self._numTrainBatches = getShape(self._trainData)[0]
        self._numTestBatches = getShape(self._testData)[0]
        self._numTestSize = self._numTestBatches * getShape(self._testData)[1]
        self._regularization = Regularization(regType, regScaleFactor)

    def __getstate__(self) :
        '''Save network pickle'''
        dict = ClassifierNetwork.__getstate__(self)
        # remove the training and test datasets before pickling. This both
        # saves disk space, and makes trained networks allow transfer learning
        if '_trainData' in dict : del dict['_trainData']
        if '_trainLabels' in dict : del dict['_trainLabels']
        if '_testData' in dict : del dict['_testData']
        if '_testLabels' in dict : del dict['_testLabels']
        if '_numTrainBatches' in dict : del dict['_numTrainBatches']
        if '_numTestBatches' in dict : del dict['_numTestBatches']
        if '_numTestSize' in dict : del dict['_numTestSize']
        # remove the functions -- they will be rebuilt JIT
        if '_checkAccuracy' in dict : del dict['_checkAccuracy']
        if '_trainNetwork' in dict : del dict['_trainNetwork']
        return dict

    def __setstate__(self, dict) :
        '''Load network pickle'''
        # remove any current functions from the object so we force the
        # theano functions to be rebuilt with the new buffers
        if hasattr(self, '_checkAccuracy') : delattr(self, '_checkAccuracy')
        if hasattr(self, '_trainNetwork') : delattr(self, '_trainNetwork')
        ClassifierNetwork.__setstate__(self, dict)

    def finalizeNetwork(self, networkInput) :
        '''Setup the network based on the current network configuration.
           This creates several network-wide functions so they will be
           pre-compiled and optimized when we need them.
        '''
        from nn.costUtils import crossEntropyLoss, compileUpdates

        if len(self._layers) == 0 :
            raise IndexError('Network must have at least one layer' +
                             'to call getNetworkInput().')

        self._startProfile('Finalizing Network', 'info')

        # disable the profiler temporarily so we don't get a second entry
        tmp = self._profiler
        self._profiler = None
        ClassifierNetwork.finalizeNetwork(self, networkInput)
        self._profiler = tmp

        # create a function to quickly check the accuracy against the test set
        index = t.lscalar('index')
        expectedLabels = t.ivector('expectedLabels')
        numCorrect = t.sum(t.eq(self._outClassMax, expectedLabels))
        # NOTE: This uses the lamda function as a means to consolidate the
        #       calling scheme. This saves us from later using conditionals in
        #       the inner loops and optimizes the libary
        if isShared(self._testData) :
            checkAcc = theano.function(
                [index], numCorrect,
                givens={self.getNetworkInput()[0] : self._testData[index],
                        expectedLabels: self._testLabels[index]})
            self._checkAccuracy = lambda ii : checkAcc(ii)
        else :
            checkAcc = theano.function(
                [self.getNetworkInput()[0], expectedLabels], numCorrect)
            self._checkAccuracy = lambda ii : checkAcc(self._testData[ii],
                                                       self._testLabels[ii])

        # create the cross entropy function --
        # This is the cost function for the network, and it assumes [0,1]
        # classification labeling. If the expectedOutput is not [0,1], Doc
        # Brown will hit you with a time machine.
        expectedOutputs = t.fmatrix('expectedOutputs') \
                          if self._trainLabels.ndim == 3 else \
                          t.ivector('expectedOutputs')
        outTrain = t.nnet.softmax(self.getNetworkOutput()[1])
        xEntropy = crossEntropyLoss(expectedOutputs, outTrain, 1)

        # create the function for back propagation of all layers --
        # weight/bias are added in reverse order because they will
        # be used back propagation, which runs output to input
        updates = compileUpdates(
            self._layers,
            (xEntropy + self._regularization.calculate(self._layers)) / \
            self.getNetworkInputSize()[0])

        # NOTE: This uses the lamda function as a means to consolidate the
        #       calling scheme. This saves us from later using conditionals in
        #       the inner loops and optimizes the libary
        if isShared(self._trainData) :
            trainNet = theano.function(
                [index], xEntropy, updates=updates,
                givens={self.getNetworkInput()[1]: self._trainData[index],
                        expectedOutputs: self._trainLabels[index]})
            self._trainNetwork = lambda ii : trainNet(ii)
        else :
            trainNet = theano.function(
                [self.getNetworkInput()[1], expectedOutputs],
                 xEntropy, updates=updates)
            self._trainNetwork = lambda ii : trainNet(self._trainData[ii],
                                                      self._trainLabels[ii])
        self._endProfile()

    def train(self, index) :
        '''Train the network against the pre-loaded inputs. This accepts
           a batch index into the pre-compiled input and expectedOutput sets.

           NOTE: Class labels for expectedOutput are assumed to be [0,1]
        '''
        self._startProfile('Training Batch [' + str(index) +
                           '/' + str(self._numTrainBatches) + ']', 'debug')
        if not hasattr(self, '_trainNetwork') :
            from dataset.shared import toShared
            inp = toShared(self._trainData[0], borrow=True) \
                  if not isShared(self._trainData) else self._trainData[0]
            self.finalizeNetwork(inp[:])
        if not isinstance(index, int) :
            raise Exception('Variable index must be an integer value')
        if index >= self._numTrainBatches :
            raise Exception('Variable index out of range for numBatches')

        # train the input --
        # the user decides if this is online or batch training
        self._trainNetwork(index)
        self._endProfile()

    def trainEpoch(self, globalEpoch, numEpochs=1) :
        '''Train the network against the pre-loaded inputs for a user-specified
           number of epochs.
           globalEpoch : total number of epochs the network has previously
                         trained
           numEpochs   : number of epochs to train this round before stopping
        '''
        for localEpoch in range(numEpochs) :
            # DEBUG: For Debugging purposes only
            #for layer in self._layers :
            #    layer.writeWeights(globalEpoch + localEpoch)
            self._startProfile('Running Epoch [' +
                               str(globalEpoch + localEpoch) + ']', 'info')
            [self.train(ii) for ii in range(self._numTrainBatches)]
            self._endProfile()
        return globalEpoch + numEpochs

    def checkAccuracy(self) :
        '''Check the accuracy against the pre-compiled the given inputs.
           This runs against the entire test set in a single call and returns
           the current accuracy of the network [0%:100%].
        '''
        self._startProfile('Checking Accuracy', 'debug')
        if not hasattr(self, '_checkAccuracy') :
            from dataset.shared import toShared
            inp = toShared(self._trainData[0], borrow=True) \
                  if not isShared(self._trainData) else self._trainData[0]
            self.finalizeNetwork(inp[:])

        # return the sum of all correctly classified targets
        acc = 0.0
        for ii in range(self._numTestBatches) :
            acc += float(self._checkAccuracy(ii))

        self._endProfile()
        return acc / float(self._numTestSize) * 100.
