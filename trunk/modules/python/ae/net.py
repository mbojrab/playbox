import theano.tensor as t
import theano
from nn.net import ClassifierNetwork
from ae.encoder import AutoEncoder
import numpy as np
from dataset.shared import isShared

class SAENetwork (ClassifierNetwork) :
    '''The SAENetwork object allows autoencoders to be stacked such that the 
       output of one autoencoder becomes the input to another. This network
       creates the necessary connections to stack the autoencoders.

       This object provides basic encoding through the classify, and
       classifyAndSoftmax functionality provided by the base class.

       filepath : Path to an already trained network on disk 
                  'None' creates randomized weighting
       prof     : Profiler to use
    '''
    def __init__ (self, filepath=None, prof=None) :
        ClassifierNetwork.__init__(self, filepath, prof)

    def addLayer(self, encoder) :
        '''Add an autoencoder to the network.'''
        if not isinstance(encoder, AutoEncoder) :
            raise TypeError('addLayer is expecting a AutoEncoder object.')
        ClassifierNetwork.addLayer(self, encoder)

    def __getstate__(self) :
        '''Save network pickle'''
        dict = ClassifierNetwork.__getstate__(self)
        # remove the functions -- they will be rebuilt JIT
        if '_encode' in dict : del dict['_encode']
        return dict

    def __setstate__(self, dict) :
        '''Load network pickle'''
        # remove any current functions from the object so we force the
        # theano functions to be rebuilt with the new buffers
        if hasattr(self, '_encode') : delattr(self, '_encode')
        ClassifierNetwork.__setstate__(self, dict)

    def finalizeNetwork(self, networkInput) :
        '''Setup the network based on the current network configuration.
           This creates several network-wide functions so they will be
           pre-compiled and optimized when we need them.
        '''
        self._startProfile('Finalizing Network', 'info')

        # disable the profiler temporarily so we don't get a second entry
        tmp = self._profiler
        self._profiler = None
        ClassifierNetwork.finalizeNetwork(self, networkInput)
        self._profiler = tmp

        self._encode = theano.function([self.getNetworkInput()[0]],
                                       self.getNetworkOutput()[0])

    def encode(self, inputs) :
        '''Encode the given inputs. The input is assumed to be 
           numpy.ndarray with dimensions specified by the first layer of the 
           network. The output is the index of the softmax classification.
        '''
        self._startProfile('Encoding the Inputs', 'debug')
        if not hasattr(self, '_encode') :
            from dataset.shared import toShared
            inp = toShared(inputs, borrow=True) \
                  if not isShared(inputs) else inputs
            self.finalizeNetwork(inp[:])

        # activating the last layer triggers all previous 
        # layers due to dependencies we've enforced
        enc = self._encode(inputs)
        self._endProfile()
        return enc

    # TODO: these should both be removed!
    def getLayer(self, layerIndex) :
        return self._layers[layerIndex]
    def writeWeights(self, layerIndex, epoch) :
        self._layers[layerIndex].writeWeights(epoch)

class ClassifierSAENetwork (SAENetwork) :
    '''The ClassifierSAENetwork adds the ability to classify a provided input
       against a target encoding feature matrix. The user must provide an
       example input(s), which the network then uses to determine similities
       between the feature data and the provided input.

       target   : Target data for network. This is the way to provide the 
                  unsupervised learning algorithm with a means to perform
                  classification.
                  NOTE: This can either be an numpy.ndarray or a path to a
                        directory of target images.
       filepath : Path to an already trained network on disk 
                  'None' creates randomized weighting
       prof     : Profiler to use
    '''
    def __init__ (self, targetData, filepath=None, prof=None) :
        SAENetwork.__init__(self, filepath, prof)

        # check if the data is currently in memory, if not read it
        self._targetData = self.__readTargetData(targetData) \
                           if isinstance(targetData, str) else \
                           targetData

    def __getstate__(self) :
        '''Save network pickle'''
        dict = SAENetwork.__getstate__(self)
        # remove the training and test datasets before pickling. This both
        # saves disk space, and makes trained networks allow transfer learning
        dict['_targetData'] = None
        if '_targetEncodings' in dict : del dict['_targetEncodings']
        # remove the functions -- they will be rebuilt JIT
        if '_closeness' in dict : del dict['_closeness']
        return dict

    def __setstate__(self, dict) :
        '''Load network pickle'''
        # remove any current functions from the object so we force the
        # theano functions to be rebuilt with the new buffers
        if hasattr(self, '_closeness') : delattr(self, '_closeness')
        # preserve the user specified softmaxTemp
        if hasattr(self, '_targetData') : 
            tmp = self._targetData
        SAENetwork.__setstate__(self, dict)
        if hasattr(self, '_targetData') : 
            self._targetData = tmp

    def __readTargetData(self, targetpath) :
        '''Read a directory of data to use as a feature matrix.'''
        import os
        from dataset.minibatch import makeContiguous
        from dataset.reader import readImage
        return makeContiguous([(readImage(os.path.join(targetpath, im))) \
                               for im in os.listdir(targetpath)])[0]

    def finalizeNetwork(self, networkInput) :
        '''Setup the network based on the current network configuration.
           This creates several network-wide functions so they will be
           pre-compiled and optimized when we need them.
        '''
        from theano import dot, function
        from dataset.shared import toShared
        import numpy as np

        if len(self._layers) == 0 :
            raise IndexError('Network must have at least one layer' +
                             'to call getNetworkInput().')

        self._startProfile('Finalizing Network', 'info')

        # disable the profiler temporarily so we don't get a second entry
        tmp = self._profiler
        self._profiler = None
        SAENetwork.finalizeNetwork(self, networkInput)
        self._profiler = tmp

        # ensure targetData is at least one batchSize, otherwise enlarge
        batchSize = networkInput.shape.eval()[0]
        numTargets = self._targetData.shape[0]
        if numTargets < batchSize :
            # add rows of zeros to fill out the rest of the batch
            self._targetData = np.resize(np.append(
                np.zeros([batchSize-numTargets] +
                         list(self._targetData.shape[1:]), 
                         np.float32), self._targetData),
                [batchSize] + list(self._targetData.shape[1:]))

        # produce the encoded feature matrix --
        # this matrix will be used for all closeness calculations
        #
        # classify the inputs one batch at a time
        enc = []
        for ii in range(int(numTargets / batchSize)) :
            enc.extend(self.encode(
                self._targetData[ii*batchSize:(ii+1)*batchSize]))

        # run one last batch and collect the remainder --
        # this is also used if there is less than one batch worth of targets
        remainder = numTargets % batchSize
        if remainder > 0 :
            enc.extend(self.encode(
                self._targetData[-batchSize:])[-remainder:])

        # reduce the encodings to only check against unique vectors --
        # this is an optimization as many examples could be encoded to
        # the same example vector.
        def uniqueRows(a):
            a = np.ascontiguousarray(a)
            unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
            return unique_a.view(a.dtype).reshape((
                unique_a.shape[0], a.shape[1]))
        enc = uniqueRows(enc)

        # NOTE: this is the transpose to orient for matrix multiplication
        self._targetEncodings = toShared(enc, borrow=True).T

        # TODO: Check if this should be the raw logit from the output layer or
        #       the softmax return of the output layer.
        # TODO: This needs to be updated to handle matrix vs matrix cosine
        #       similarities between all pairs of vectors
        # setup the closeness execution graph based on target information
        targets = t.fmatrix('targets')
        outClass = self.getNetworkOutput()[0]
        cosineSimilarity = dot(outClass, targets) / \
            (t.sqrt(t.sum(outClass**2)) * (t.sqrt(t.sum(targets**2))))
        self._closeness = function([self.getNetworkInput()[0]],
                                   t.mean(cosineSimilarity, axis=1),
                                   givens={targets: self._targetEncodings})
        self._endProfile()

    def closeness(self, inputs, cosineVector=None) :
        '''This is a form of classification for SAE networks. The network has
           been provided a target input, which we now use to determine the
           similarity of this input against that target set. 

           inputs:       Example imagery to test for closeness. 
                         (batchSize, numChannels, rows, cols)
           cosineVector: Pre-initialized vector. Use this when the input needs
                         to be biased, or if you are normalizing the responses
                         from several networks.

           return      : The calculation returns a value between [0., 1.] for
                         each input. If the user specifies a cosineVector, the
                         responses from this network are added to the previous
                         vector. If cosineVector is None, the networks raw 
                         responses are returned.

           NOTE: Response of 1.0 indicates equality. The lower number indicate
                 less overlap between features.
        '''
        from dataset.shared import toShared
        self._startProfile('Determining Closeness of Inputs', 'debug')
        if not hasattr(self, '_closeness') :
            inp = toShared(inputs, borrow=True) \
                  if not isShared(inputs) else inputs
            self.finalizeNetwork(inp[:])
        if not hasattr(self, '_targetEncodings') :
            raise ValueError('User must finalize the feature matrix before ' +
                             'attempting to finalize the network.')

        # test out similar this input is compared with the targets
        if cosineVector is not None :
            cosineVector += self._closeness(inputs)
        else :
            cosineVector = self._closeness(inputs)
        self._endProfile()
        return cosineVector

    def closenessAndEncoding (self, inputs) :
        '''This is a form of classification for SAE networks. The network has
           been provided a target input, which we now use to determine the
           similarity of this input against that target set.

           This method is additionally setup to return the raw encoding for the
           inputs provided.

           return : (classification index, softmax vector)
        '''
        self._startProfile('Determining Closeness of Inputs', 'debug')

        # TODO: this needs to be updated if the encodings should not be the
        #       result of a softmax on the logits.
        cosineMatrix, encodings = (self.closeness(inputs),
                                   self.classifyAndSoftmax(inputs)[1])
        self._endProfile()
        return cosineMatrix, encodings


class TrainerSAENetwork (SAENetwork) :
    '''The TrainerSAENetwork object expands on the classification and allows
       allows training of the stacked autoencoder both in a greedy-layerwise
       and network-wide manner. 

       NOTE: The resulting trained AEs can be used to initialize a 
             nn.TrainerNetwork.

       train    : theano.shared dataset used for network training in format --
                  (numBatches, batchSize, numChannels, rows, cols)
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
    def __init__ (self, train, regType='L2', regScaleFactor=0.,
                  filepath=None, prof=None) :
        from nn.reg import Regularization
        SAENetwork.__init__ (self, filepath, prof)
        self._indexVar = t.lscalar('index')
        self._trainData = train[0] if isinstance(train, list) else train
        self._numTrainBatches = self._trainData.shape.eval()[0]
        self._trainGreedy = []
        self._regularization = Regularization(regType, regScaleFactor)

    def __buildEncoder(self) :
        '''Build the greedy-layerwise function --
           All layers start with the input original input, however are updated
           in a layerwise manner.
           NOTE: this uses theano.shared variables for optimized GPU execution
        '''
        for encoder in self._layers :
            # forward pass through layers
            self._startProfile('Finalizing Encoder [' + encoder.layerID + ']', 
                               'debug')
            out, up = encoder.getUpdates()
            self._trainGreedy.append(
                theano.function([self._indexVar], out, updates=up,
                                givens={self.getNetworkInput()[0] : 
                                        self._trainData[self._indexVar]}))
            self._endProfile()

    def __buildDecoder(self) :
        '''Build the decoding section and the network-wide training method.'''
        from nn.costUtils import calcLoss, \
                                 calcSparsityConstraint, \
                                 compileUpdates

        # setup the decoders -- 
        # this is the second half of the network and is equivalent to the
        # encoder network reversed.
        layerInput = self.getNetworkOutput()[0]
        sparseConstr = calcSparsityConstraint(
            layerInput, self.getNetworkOutputSize())
        jacobianCost = self._layers[-1].getUpdates()[0][1]

        for decoder in reversed(self._layers) :
            # backward pass through layers
            self._startProfile('Finalizing Decoder [' + decoder.layerID + ']', 
                               'debug')
            layerInput = decoder.buildDecoder(layerInput)
            self._endProfile()
        decodedInput = layerInput

        self.reconstruction = theano.function([self.getNetworkInput()[0]],
                                              decodedInput)

        # TODO: here we assume the first layer uses sigmoid activation
        self._startProfile('Setting up Network-wide Decoder', 'debug')

        netInput = self.getNetworkInput()[0].flatten(2) \
                   if len(self.getNetworkInput()[0].shape.eval()) != \
                      len(self.getNetworkInputSize()) else \
                   self.getNetworkInput()[0]
        cost = calcLoss(netInput, decodedInput,
                        self._layers[0].getActivation()) / \
                        self.getNetworkInputSize()[0]
        costs = [cost, jacobianCost, sparseConstr]
        costs.extend(x for x in [self._regularization.calculate(self._layers)]\
                     if x is not None)

        # build the network-wide training update.
        updates = compileUpdates(self._layers, t.sum(costs))

        #from theano.compile.nanguardmode import NanGuardMode
        self._trainNetwork = theano.function(
            [self._indexVar], costs, updates=updates, 
            givens={self.getNetworkInput()[0] : 
                    self._trainData[self._indexVar]})
            #mode=NanGuardMode(nan_is_error=True, inf_is_error=True,\
            #                   big_is_error=True))
        self._endProfile()

    def __getstate__(self) :
        '''Save network pickle'''
        dict = SAENetwork.__getstate__(self)
        # remove the functions -- they will be rebuilt JIT
        if '_indexVar' in dict : del dict['_indexVar']
        if '_trainData' in dict : del dict['_trainData']
        if '_numTrainBatches' in dict : del dict['_numTrainBatches']
        if '_trainGreedy' in dict : del dict['_trainGreedy']
        if '_trainNetwork' in dict : del dict['_trainNetwork']
        if 'reconstruction' in dict : del dict['reconstruction']
        return dict

    def __setstate__(self, dict) :
        '''Load network pickle'''
        self._indexVar = t.lscalar('index')
        # remove any current functions from the object so we force the
        # theano functions to be rebuilt with the new buffers
        if hasattr(self, '_trainGreedy') : delattr(self, '_trainGreedy')
        if hasattr(self, '_trainNetwork') : delattr(self, '_trainNetwork')
        if hasattr(self, 'reconstruction') : delattr(self, 'reconstruction')
        self._trainGreedy = []
        SAENetwork.__setstate__(self, dict)

    def finalizeNetwork(self, networkInputs) :
        '''Setup the network based on the current network configuration.
           This is used to create several network-wide functions so they will
           be pre-compiled and optimized when we need them. The only function
           across all network types is classify()
        '''
        if len(self._layers) == 0 :
            raise IndexError('Network must have at least one layer' +
                             'to call finalizeNetwork().')

        self._startProfile('Finalizing Network', 'info')

        # disable the profiler temporarily so we don't get a second entry
        tmp = self._profiler
        self._profiler = None
        SAENetwork.finalizeNetwork(self, networkInputs)
        self._profiler = tmp

        self.__buildEncoder()
        self.__buildDecoder()
        self._endProfile()

    def train(self, layerIndex, index) :
        '''Train the network against the pre-loaded inputs. This accepts
           a batch index into the pre-compiled input set.
           layerIndex : specify which layer to train
           index      : specify a pre-compiled mini-batch index
           inputs     : DEBUGGING Specify a numpy tensor mini-batch
        '''
        self._startProfile('Training Batch [' + str(index) +
                           '/' + str(self._numTrainBatches) + ']', 'debug')
        if not hasattr(self, '_trainGreedy') or \
           not hasattr(self, '_trainNetwork') :
            self.finalizeNetwork(self._trainData[0])
        if not isinstance(index, int) :
            raise Exception('Variable index must be an integer value')
        if index >= self._numTrainBatches :
            raise Exception('Variable index out of range for numBatches')

        # train the input --
        # the user decides whether this will be a greedy or network training
        # by passing in a layer index. If the index does not have an associated
        # layer, it automatically chooses network-wide training.
        if layerIndex < 0 or layerIndex >= self.getNumLayers() :
            ret = self._trainNetwork(index)
        else :
            ret = self._trainGreedy[layerIndex](index)

        self._endProfile()
        return ret

    def trainEpoch(self, layerIndex, globalEpoch, numEpochs=1) :
        '''Train the network against the pre-loaded inputs for a user-specified
           number of epochs.

           layerIndex  : index of the layer to train
           globalEpoch : total number of epochs the network has previously 
                         trained
           numEpochs   : number of epochs to train this round before stopping
        '''
        globCost = []
        for localEpoch in range(numEpochs) :
            layerEpochStr = 'Layer[' + str(layerIndex) + '] Epoch[' + \
                            str(globalEpoch + localEpoch) + ']'
            self._startProfile('Running ' + layerEpochStr, 'info')
            locCost = []
            for ii in range(self._numTrainBatches) :
                locCost.append(self.train(layerIndex, ii))

            # log the cost
            locCost = np.mean(locCost, axis=0)
            costMessage = layerEpochStr + ' Cost: ' + str(locCost[0])
            if len(locCost) >= 2 :
                costMessage += ' - Jacob: ' + str(locCost[1])
            if len(locCost) >= 3 :
                costMessage += ' - Sparsity: ' + str(locCost[2])
            if len(locCost) == 4 :
                costMessage += ' - Regularization: ' + str(locCost[3])
            self._startProfile(costMessage, 'info')
            globCost.append(locCost)
            self._endProfile()

            self._endProfile()

            '''            # DEBUG: For debugging pursposes only!
            from dataset.debugger import saveTiledImage
            if layerIndex >= 0 and layerIndex < self.getNumLayers() :
                reconstructedInput = self._layers[layerIndex].reconstruction(
                    self._trainData.get_value(borrow=True)[0])
                reconstructedInput = np.resize(
                    reconstructedInput, 
                    self._layers[layerIndex].getInputSize())
                tileShape = None
                if layerIndex == 0 :
                    imageShape = (28,28)
                    reconstructedInput = np.resize(reconstructedInput,
                                                   (50,1,28,28))
                elif len(self._layers[layerIndex].getInputSize()) > 2 :
                    imageShape = tuple(self._layers[
                                       layerIndex].getInputSize()[-2:])
                else :
                    imageShape=(1, self._layers[layerIndex].getInputSize()[1])
                    tileShape=(self._layers[layerIndex].getInputSize()[0], 1)

                self.writeWeights(layerIndex, globalEpoch + localEpoch)
                saveTiledImage(image=reconstructedInput,
                               path=self._layers[layerIndex].layerID +
                                    '_reconstruction_' + 
                                    str(globalEpoch+localEpoch) + '.png',
                               imageShape=imageShape, tileShape=tileShape,
                               spacing=1, interleave=True)
            else :
                reconstructedInput = self.reconstruction(
                    self._trainData.get_value(borrow=True)[0])
                imageShape = (28,28)
                reconstructedInput = np.resize(reconstructedInput, 
                                               (50,1,28,28))
                saveTiledImage(image=reconstructedInput,
                               path='network_reconstruction_' + 
                                     str(globalEpoch+localEpoch) + '.png',
                               imageShape=imageShape, spacing=1,
                               interleave=True)
            '''
        return globalEpoch + numEpochs, globCost
