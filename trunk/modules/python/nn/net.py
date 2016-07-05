from nn.layer import Layer
import theano.tensor as t
import theano
import numpy as np
from dataset.pickle import writePickleZip, readPickleZip

class Network () :
    def __init__ (self, prof=None) :
        self._profiler = prof
        self._layers = []
    def __getstate__(self) :
        '''Save network pickle'''
        dict = self.__dict__.copy()
        # remove the profiler as it is not robust to distributed processing
        dict['_profiler'] = None
        return dict
    def __setstate__(self, dict) :
        '''Load network pickle'''
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
        if '.pkl.gz' in filepath :
            self.__setstate__(readPickleZip(filepath))
        self._endProfile()
    def getNumLayers(self) :
        return len(self._layers)
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

class ClassifierNetwork (Network) :
    '''The ClassifierNetwork object allows the user to build multi-layer neural
       networks of various topologies easily. This class provides users with 
       functionality to load a trained Network from disk and begin classifying
       inputs. 

       filepath : Path to an already trained network on disk 
                  'None' creates randomized weighting
       prof     : Profiler to use
    '''
    def __init__ (self, filepath=None, prof=None) :
        Network.__init__(self, prof)
        self._networkLabels = []
        if filepath is not None :
            self.load(filepath)

    def __getstate__(self) :
        '''Save network pickle'''
        dict = Network.__getstate__(self)
        # remove the functions -- they will be rebuilt JIT
        if '_outClassSoft' in dict : del dict['_outClassSoft']
        if '_outTrainSoft' in dict : del dict['_outTrainSoft']
        if '_outClassMax' in dict : del dict['_outClassMax']
        if '_classify' in dict : del dict['_classify']
        if '_classifyAndSoftmax' in dict : del dict['_classifyAndSoftmax']
        return dict
    def __setstate__(self, dict) :
        '''Load network pickle'''
        # remove any current functions from the object so we force the
        # theano functions to be rebuilt with the new buffers
        if hasattr(self, '_outClassSoft') : 
            delattr(self, '_outClassSoft')
        if hasattr(self, '_outTrainSoft') : 
            delattr(self, '_outTrainSoft')
        if hasattr(self, '_outClassMax') : 
            delattr(self, '_outClassMax')
        if hasattr(self, '_classify') : 
            delattr(self, '_classify')
        if hasattr(self, '_classifyAndSoftmax') : 
            delattr(self, '_classifyAndSoftmax')
        Network.__setstate__(self, dict)

    def getNetworkLabels(self) :
        '''Return the Labels for the network. All other interactions with
           training and accuracy deal with the label index, so this decodes
           it into a string classification.
        '''
        return self._networkLabels

    def convertToLabels(self, labelIndices) :
        '''Return the string labels for a vector of indices.'''
        return [self._networkLabels[ii] for ii in self._networkLabels]

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
        # Here we use softmax on the network output to produce a normalized 
        # output prediction, which emphasizes significant neural responses.
        # This takes as its input, the first layer's input, and uses the final
        # layer's output as the function (ie the network classification).
        outClass, outTrain = self.getNetworkOutput()
        self._outClassSoft = t.nnet.softmax(outClass)
        self._outTrainSoft = t.nnet.softmax(outTrain)
        self._outClassMax = t.argmax(self._outClassSoft, axis=1)
        self._classify = theano.function([self.getNetworkInput()[0]], 
                                         self._outClassMax)
        self._classifyAndSoftmax = theano.function(
            [self.getNetworkInput()[0]], 
            [self._outClassMax, self._outClassSoft])
        self._endProfile()

    def classify (self, inputs) :
        '''Classify the given inputs. The input is assumed to be 
           numpy.ndarray with dimensions specified by the first layer of the 
           network. The output is the index of the softmax classification.
        '''
        self._startProfile('Classifying the Inputs', 'debug')
        if not hasattr(self, '_classify') :
            self.finalizeNetwork()

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
            self.finalizeNetwork()

        # activating the last layer triggers all previous 
        # layers due to dependencies we've enforced
        classIndex, softmax = self._classifyAndSoftmax(inputs)
        self._endProfile()
        return classIndex, softmax

class TrainerNetwork (ClassifierNetwork) :
    '''This network allows for training data on a theano.shared wrapped
       dataset for optimal execution. Because the dataset will be accessed 
       repetitively over the course of training, the shared variables are
       preloaded unto the target architecture. The input to training will be
       an index into this array.

       The network uses a softmax normalization on the output vector to 
       obtain [0,1] classification. This allows for a cross-entropy (ie nll)
       loss function.

       train    : theano.shared dataset used for network training in format
                  NOTE: Currently the user is allowed to pass two variable
                        types for this field. --

                        If the user is passing an index
                        equivalent for the label, the user must pass data as a
                        numpy.ndarray and formatted:

                        (((numBatches, batchSize, numChannels, rows, cols)), 
                         (numBatches, oneHotIndex))

                        If the user passes a vector for the expected label the
                        values must be theano.shared variables and formatted:

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
    '''
    def __init__ (self, train, test, labels, regType='L2', regScaleFactor=0.,
                  filepath=None, prof=None) :
        ClassifierNetwork.__init__(self, filepath=filepath, prof=prof)
        if filepath is None :
            self._networkLabels = labels
        self._trainData, self._trainLabels = train
        self._testData, self._testLabels = test

        self._numTrainBatches = self._trainLabels.shape.eval()[0]
        self._numTestBatches = self._testLabels.shape.eval()[0]
        self._numTestSize = self._numTestBatches * \
                            self._testLabels.shape.eval()[1]
        self._regularization = regType
        self._regScaleFactor = regScaleFactor

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
        if '_regularization' in dict : del dict['_regularization']
        # remove the functions -- they will be rebuilt JIT
        if '_checkAccuracy' in dict : del dict['_checkAccuracy']
        if '_createBatchExpectedOutput' in dict :
            del dict['_createBatchExpectedOutput']
        if '_cost' in dict : del dict['_cost']
        if '_trainNetwork' in dict : del dict['_trainNetwork']
        return dict

    def __setstate__(self, dict) :
        '''Load network pickle'''
        # remove any current functions from the object so we force the
        # theano functions to be rebuilt with the new buffers
        if hasattr(self, '_checkAccuracy') :
            delattr(self, '_checkAccuracy')
        if hasattr(self, '_createBatchExpectedOutput') :
            delattr(self, '_createBatchExpectedOutput')
        if hasattr(self, '_cost') : delattr(self, '_cost')
        if hasattr(self, '_trainNetwork') : delattr(self, '_trainNetwork')
        ClassifierNetwork.__setstate__(self, dict)

    def finalizeNetwork(self) :
        '''Setup the network based on the current network configuration.
           This creates several network-wide functions so they will be
           pre-compiled and optimized when we need them.
        '''
        from nn.costUtils import crossEntropyLoss
        from nn.costUtils import leastAbsoluteDeviation, leastSquares
        if len(self._layers) == 0 :
            raise IndexError('Network must have at least one layer' +
                             'to call getNetworkInput().')

        self._startProfile('Finalizing Network', 'info')

        # finalize the layers to create the computational graphs
        layerInput = (self._trainData, self._trainData)
        for layer in self._layers :
            layer.finalize(layerInput)
            layerInput = layer.output

        # disable the profiler temporarily so we don't get a second entry
        tmp = self._profiler
        self._profiler = None
        ClassifierNetwork.finalizeNetwork(self)
        self._profiler = tmp

        # create a function to quickly check the accuracy against the test set
        index = t.lscalar('index')
        expectedLabels = t.ivector('expectedLabels')
        numCorrect = t.sum(t.eq(self._outClassMax, expectedLabels))
        # NOTE: the 'input' variable name was created elsewhere and provided as
        #       input to the first layer. We now use that object to connect
        #       our shared buffers.
        self._checkAccuracy = theano.function(
            [index], numCorrect, 
            givens={self.getNetworkInput()[0] : self._testData[index],
                    expectedLabels: self._testLabels[index]})

        # create the cross entropy function --
        # This is the cost function for the network, and it assumes [0,1]
        # classification labeling. If the expectedOutput is not [0,1], Doc
        # Brown will hit you with a time machine.
        expectedOutputs = t.fmatrix('expectedOutputs') \
                          if self._trainLabels.ndim == 3 else \
                          t.ivector('expectedOutputs')
        xEntropy = crossEntropyLoss(expectedOutputs, self._outTrainSoft, 1)


        # calculate a regularization term -- if desired
        reg = 0.0
        # L1-norm provides 'Least Absolute Deviation' --
        # built for sparse outputs and is resistent to outliers
        if self._regularization == 'L1' :
            reg = leastAbsoluteDeviation(
                [l.getWeights()[0] for l in self._layers], 
                self._regScaleFactor)
        # L2-norm provides 'Least Squares' --
        # built for dense outputs and is computationally stable at small errors
        elif self._regularization == 'L2' :
            reg = leastSquares(
                [l.getWeights()[0] for l in self._layers],
                self._regScaleFactor)


        # create the function for back propagation of all layers --
        # weight/bias are added in reverse order because they will
        # be used back propagation, which runs output to input
        updates = []
        for layer in reversed(self._layers) :

            # pull the rate variables
            layerLearningRate = layer.getLearningRate()
            layerMomentumRate = layer.getMomentumRate()

            # build the gradients
            layerWeights = layer.getWeights()
            gradients = t.grad(xEntropy + reg, layerWeights,
                               disconnected_inputs='warn')

            # add the weight update
            for w, g in zip(layerWeights, gradients) :

                if layerMomentumRate > 0. :
                    # setup a second buffer for storing momentum
                    previousWeightUpdate = theano.shared(
                        np.zeros(w.get_value().shape, theano.config.floatX),
                        borrow=True)

                    # add two updates --
                    # perform weight update and save the previous update
                    updates.append((w, w + previousWeightUpdate))
                    updates.append((previousWeightUpdate,
                                    previousWeightUpdate * layerMomentumRate -
                                    layerLearningRate * g))
                else :
                    updates.append((w, w - layerLearningRate * g))


        # NOTE: the 'input' variable name was create elsewhere and provided as
        #       input to the first layer. We now use that object to connect
        #       our shared buffers.
        # NOTE: This check should only be used until both buffers are in 
        self._trainNetwork = theano.function(
            [index], xEntropy, updates=updates,
            givens={self.getNetworkInput()[1]: self._trainData[index],
                    expectedOutputs: self._trainLabels[index]})
        self._endProfile()

    #def train(self, index) :
    def train(self, index) :
        '''Train the network against the pre-loaded inputs. This accepts 
           a batch index into the pre-compiled input and expectedOutput sets.

           NOTE: Class labels for expectedOutput are assumed to be [0,1]
        '''
        self._startProfile('Training Batch [' + str(index) +
                           '/' + str(self._numTrainBatches) + ']', 'debug')
        if not hasattr(self, '_trainNetwork') :
            self.finalizeNetwork()
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
            self.finalizeNetwork()

        # return the sum of all correctly classified targets
        acc = 0.0
        for ii in range(self._numTestBatches) :
            acc += float(self._checkAccuracy(ii))

        self._endProfile()
        return acc / float(self._numTestSize) * 100.
