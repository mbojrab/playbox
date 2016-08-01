import theano.tensor as t
import theano
from nn.net import TrainerNetwork, ClassifierNetwork

class DistilleryClassifier(ClassifierNetwork) :
    '''The ClassifierNetwork object allows the user to build multi-layer neural
       networks of various topologies easily. This class provides users with 
       functionality to load a trained Network from disk and begin classifying
       inputs. 

       filepath    : Path to an already trained network on disk 
                     'None' creates randomized weighting
       softmaxTemp : Temperature for the softmax method. A larger value softens
                     the output from softmax. A value of 1.0 return a standard
                     softmax result.
       prof        : Profiler to use
    '''
    def __init__ (self, filepath=None, softmaxTemp=4., prof=None) :
        ClassifierNetwork.__init__(self, filepath, prof)
        self._softmaxTemp = softmaxTemp

    def __getstate__(self) :
        '''Save network pickle'''
        dict = ClassifierNetwork.__getstate__(self)
        # always use the most recent one specified by the user.
        dict['_softmaxTemp'] = None

        # remove the training and test datasets before pickling. This both
        # saves disk space, and makes trained networks allow transfer learning
        if '_softClassify' in dict : del dict['_softClassify']
        return dict

    def __setstate__(self, dict) :
        '''Load network pickle'''
        # remove any current functions from the object so we force the
        # theano functions to be rebuilt with the new buffers
        if hasattr(self, '_softTarget') : delattr(self, '_softTarget')

        # preserve the user specified entries
        if hasattr(self, '_softmaxTemp') : 
            tmpTemp = self._softmaxTemp
        ClassifierNetwork.__setstate__(self, dict)
        if hasattr(self, '_softmaxTemp') : 
            self._softmaxTemp = tmpTemp

    def softTarget(self, inputs) :
        '''The output the soft target from the network. '''
        self._startProfile('Classifying the Inputs', 'debug')
        if not hasattr(self, '_softTarget') :
            from dataset.shared import toShared
            inp = toShared(inputs, borrow=True) \
                  if 'SharedVariable' not in str(type(inputs)) else inputs
            self.finalizeNetwork(inp[:])

        # activating the last layer triggers all previous 
        # layers due to dependencies we've enforced
        softTarget = self._softTarget(inputs)
        self._endProfile()
        return softTarget

    def finalizeNetwork(self, networkInputs) :
        '''Setup the network based on the current network configuration.
           This creates several network-wide functions so they will be
           pre-compiled and optimized when we need them.
        '''
        from nn.probUtils import softmaxAction

        self._startProfile('Finalizing Network', 'info')

        # disable the profiler temporarily so we don't get a second entry
        tmp = self._profiler
        self._profiler = None
        ClassifierNetwork.finalizeNetwork(self, networkInputs)
        self._profiler = tmp

        # setup a new classify function to output the soft targets
        softClass = softmaxAction(self.getNetworkOutput()[0], 
                                  self._softmaxTemp)
        self._softTarget = theano.function([self.getNetworkInput()[0]],
                                             softClass)


class DistilleryTrainer (TrainerNetwork) :
    '''This network allows distillation from one network to another. 

       train       : theano.shared dataset used for network training in format
                     NOTE: Currently the user is allowed to pass two variable
                           types for this field. --

                           If the user is passing an index
                           equivalent for the label, the user must pass data as
                           a numpy.ndarray and formatted:

                           (((numBatches, batchSize, numChannels, rows, cols)), 
                            (numBatches, oneHotIndex))

                           If the user passes a vector for the expected label
                           the values must be theano.shared variables and
                           formatted:

                           (((numBatches, batchSize, numChannels, rows, cols)), 
                            (numBatches, batchSize, expectedOutputVect))

       test        : theano.shared dataset used for network testing in format--
                     (((numBatches, batchSize, numChannels, rows, cols)), 
                     integerLabelIndices)
                     The intersection of train and test datasets should be a
                     null set. The test dataset will be used to regularize the
                     training
       regType     : type of regularization term to use
                     default None : perform no additional regularization
                     L1           : Least Absolute Deviation
                     L2           : Least Squares
       regSF       : regularization scale factor
                     NOTE: a good value is 1. / numTotalNeurons
       filepath    : Path to an already trained network on disk 
                     'None' creates randomized weighting
       softmaxTemp : Temperature for the softmax method. A larger value softens
                     the output from softmax. A value of 1.0 return a standard
                     softmax result.

       prof     : Profiler to use
    '''
    def __init__ (self, train, test, labels, regType='L2', regScaleFactor=0.,
                  filepath=None, softmaxTemp=4., transFactor=0.8, prof=None) :
        TrainerNetwork.__init__(self, train[:2], test, labels, regType, 
                                regScaleFactor, filepath, prof)
        self._trainKnowledge = train[2] if len(train) > 2 else None
        self._softmaxTemp = softmaxTemp
        self._transFactor = transFactor

    def __getstate__(self) :
        '''Save network pickle'''
        dict = TrainerNetwork.__getstate__(self)
        # always use the most recent one specified by the user.
        if '_softmaxTemp' in dict : del dict['_softmaxTemp']
        if '_transFactor' in dict : del dict['_transFactor']

        # remove the training and test datasets before pickling. This both
        # saves disk space, and makes trained networks allow transfer learning
        if '_trainKnowledge' in dict : del dict['_trainKnowledge']
        if '_trainNetwork' in dict : del dict['_trainNetwork']
        if '_deepNet' in dict : del dict['_deepNet']
        return dict

    def __setstate__(self, dict) :
        '''Load network pickle'''
        # remove any current functions from the object so we force the
        # theano functions to be rebuilt with the new buffers
        if hasattr(self, '_trainKnowledge') : delattr(self, '_trainKnowledge')
        if hasattr(self, '_trainNetwork') : delattr(self, '_trainNetwork')

        # preserve the user specified entries
        if hasattr(self, '_deepNet') : 
            tmpDeep = self._deepNet
        if hasattr(self, '_softmaxTemp') : 
            tmpTemp = self._softmaxTemp
        if hasattr(self, '_transFactor') : 
            tmpFactor = self._transFactor
        TrainerNetwork.__setstate__(self, dict)
        if hasattr(self, '_deepNet') : 
            self._deepNet = tmpDeep
        if hasattr(self, '_softmaxTemp') : 
            self._softmaxTemp = tmpTemp
        if hasattr(self, '_transFactor') : 
            self._transFactor = tmpFactor

    def loadDeepNetwork(self, filepath) :
        '''Load a network into memory to use it as the target for distillation.
        '''
        if self._trainKnowledge is not None :
            raise ValueError('A soft target source was already specified. ' + 
                             'Only one source should be specified.')
        self._deepNet = DistilleryClassifier(filepath, self._softmaxTemp,
                                             self._profiler)
        self._deepNet.finalizeNetwork(self._trainData[0])

    def finalizeNetwork(self, networkInput) :
        '''Setup the network based on the current network configuration.
           This creates several network-wide functions so they will be
           pre-compiled and optimized when we need them.
        '''
        from nn.probUtils import softmaxAction
        from nn.costUtils import crossEntropyLoss

        if not hasattr(self, '_deepNet') and self._trainKnowledge is None :
            raise ValueError('Please either specify soft targets either ' +
                             'through the train object or load a deep network.')

        self._startProfile('Finalizing Network', 'info')

        # disable the profiler temporarily so we don't get a second entry
        tmp = self._profiler
        self._profiler = None
        TrainerNetwork.finalizeNetwork(self, networkInput)
        self._profiler = tmp

        # setup the training to use soft targets from the deep network --
        # NOTE: This is different from the basic training in that it uses both
        #       the hard targets as well as those provided by a deep network.
        #       The deep network provides supercharged information about
        #       similarities between categories for the inputs, and in the
        #       cases where the deep network is incorrect, the hard targets
        #       help the network to update itself on the side of correctness.
        #
        # setup for knowledge transfer
        index = t.lscalar('index')
        softOut = softmaxAction(self.getNetworkOutput()[1], self._softmaxTemp)
        if hasattr(self, '_deepNet') :
            # here we use the deepNet to obtain the soft targets JIT
            deepExpect = softmaxAction(self._deepNet.getNetworkOutput()[1],
                                       self._softmaxTemp)
            etcGivens = {self._deepNet.getNetworkInput()[1]: 
                         self._trainData[index]}
        else :
            # here we use the soft targets loaded from the pickle
            deepExpect = t.fmatrix('deepExpect')
            etcGivens = {deepExpect: self._trainKnowledge[index]}
        deepXEntropy = crossEntropyLoss(deepExpect, softOut, 1)

        # setup the hard targets appropriately --
        # this should use the temp of 1.0 to signify a regular softmax
        hardExpect = t.ivector('hardExpect')
        hardOut = t.nnet.softmax(self.getNetworkOutput()[1])
        hardXEntropy = crossEntropyLoss(hardExpect, hardOut, 1)

        # create the function for back propagation of all layers --
        # weight/bias are added in reverse order because they will
        # be used back propagation, which runs output to input
        updates = self._compileUpdates(
            (self._transFactor * deepXEntropy) +
            (1. - self._transFactor) * hardXEntropy +
            self._compileRegularization())

        # override the training function
        givens = {self.getNetworkInput()[1]: self._trainData[index],
                  hardExpect: self._trainLabels[index]}
        givens.update(etcGivens)
        self._trainNetwork = theano.function(
            [index], [deepXEntropy, hardXEntropy], updates=updates,
            givens=givens)
        self._endProfile()
