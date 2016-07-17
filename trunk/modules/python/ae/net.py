import theano.tensor as t
import theano
from nn.net import ClassifierNetwork
from ae.encoder import AutoEncoder
import numpy as np

class ClassifierSAENetwork (ClassifierNetwork) :
    '''The ClassifierSAENetwork object allows autoencoders to be stacked such
       that the output of one autoencoder becomes the input to another. This 
       network creates the necessary connections to stack the autoencoders.

       filepath    : Path to an already trained network on disk 
                     'None' creates randomized weighting
       softmaxTemp : Temperature for the softmax method. A larger value softens
                     the output from softmax. A value of 1.0 return a standard
                     softmax result.
       prof        : Profiler to use
    '''
    def __init__ (self, filepath=None, softmaxTemp=0., prof=None) :
        ClassifierNetwork.__init__(self, filepath, softmaxTemp, prof)

    def addLayer(self, encoder) :
        '''Add an autoencoder to the network.'''
        if not isinstance(encoder, AutoEncoder) :
            raise TypeError('addLayer is expecting a AutoEncoder object.')
        ClassifierNetwork.addLayer(self, encoder)

    # TODO: these should both be removed!
    def getLayer(self, layerIndex) :
        return self._layers[layerIndex]
    def writeWeights(self, layerIndex, epoch) :
        self._layers[layerIndex].writeWeights(epoch)


class TrainerSAENetwork (ClassifierSAENetwork) :
    '''The TrainerSAENetwork object expands on the classification and allows
       allows training of the stacked autoencoder both in a greedy-layerwise
       and network-wide manner. 

       NOTE: The resulting trained AEs can be used to initialize a 
             nn.TrainerNetwork.

       train : theano.shared dataset used for network training in format --
               (numBatches, batchSize, numChannels, rows, cols)
       prof  : Profiler to use
    '''
    def __init__ (self, train, filepath=None, softmaxTemp=0., prof=None) :
        ClassifierSAENetwork.__init__ (self, filepath, softmaxTemp, prof)
        self._indexVar = t.lscalar('index')
        self._trainData = train[0] if isinstance(train, tuple) else train
        self._numTrainBatches = self._trainData.shape.eval()[0]
        self._trainGreedy = []

    def __buildEncoder(self) :
        '''Build the greedy-layerwise function --
           All layers start with the input original input, however are updated
           in a layerwise manner.
           NOTE: this uses theano.shared variables for optimized GPU execution
        '''
        layerInput = (self._trainData[0], self._trainData[0])
        for encoder in self._layers :
            # forward pass through layers
            self._startProfile('Finalizing Encoder [' + encoder.layerID + ']', 
                               'debug')
            encoder.finalize(layerInput)
            out, up = encoder.getUpdates()
            self._trainGreedy.append(
                theano.function([self._indexVar], out, updates=up,
                                givens={self.getNetworkInput()[1] : 
                                        self._trainData[self._indexVar]}))
            layerInput = encoder.output
            self._endProfile()

    def __buildDecoder(self) :
        '''Build the decoding section and the network-wide training method.'''
        from nn.costUtils import calcLoss, calcSparsityConstraint

        # setup the decoders -- 
        # this is the second half of the network and is equivalent to the
        # encoder network reversed.
        layerInput = self._layers[-1].output[1]
        sparseConstr = calcSparsityConstraint(layerInput, 
                                              self.getNetworkOutputSize())

        jacobianCost = 0
        for decoder in reversed(self._layers) :
            # backward pass through layers
            self._startProfile('Finalizing Decoder [' + decoder.layerID + ']', 
                               'debug')
            jacobianCost += decoder.getUpdates()[0][1]
            layerInput = decoder.buildDecoder(layerInput)
            self._endProfile()
        decodedInput = layerInput

        # TODO: here we assume the first layer uses sigmoid activation
        self._startProfile('Setting up Network-wide Decoder', 'debug')
        cost = calcLoss(self._trainData[0], decodedInput, t.nnet.sigmoid)
        costs = [cost, jacobianCost, sparseConstr]

        # build the network-wide training update. 
        updates = []
        for decoder in reversed(self._layers) :
            
            # build the gradients
            layerWeights = decoder.getWeights()
            gradients = t.grad(t.sum(costs), layerWeights, 
                               disconnected_inputs='warn')

            # add the weight update
            for w, g in zip(layerWeights, gradients) :
                updates.append((w, w - decoder.getLearningRate() * g))

        self._trainNetwork = theano.function(
            [self._indexVar], costs, updates=updates, 
            givens={self.getNetworkInput()[1] : 
                    self._trainData[self._indexVar]})
        self._endProfile()

    def __getstate__(self) :
        '''Save network pickle'''
        dict = ClassifierSAENetwork.__getstate__(self)
        # remove the functions -- they will be rebuilt JIT
        if '_indexVar' in dict : del dict['_indexVar']
        if '_trainData' in dict : del dict['_trainData']
        if '_numTrainBatches' in dict : del dict['_numTrainBatches']
        if '_trainGreedy' in dict : del dict['_trainGreedy']
        if '_trainNetwork' in dict : del dict['_trainNetwork']
        return dict

    def __setstate__(self, dict) :
        '''Load network pickle'''
        self._indexVar = t.lscalar('index')
        # remove any current functions from the object so we force the
        # theano functions to be rebuilt with the new buffers
        if hasattr(self, '_trainGreedy') : delattr(self, '_trainGreedy')
        if hasattr(self, '_trainNetwork') : delattr(self, '_trainNetwork')
        self._trainGreedy = []
        ClassifierSAENetwork.__setstate__(self, dict)

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

        self.__buildEncoder()

        # disable the profiler temporarily so we don't get a second entry
        tmp = self._profiler
        self._profiler = None
        ClassifierNetwork.finalizeNetwork(self)
        self._profiler = tmp

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
            self.finalizeNetwork()
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
            '''
            reconstructedInput = self._layers[layerIndex].reconstruction(
                                    self._trainData.get_value(borrow=True)[0])
            from dataset.debugger import saveTiledImage
            saveTiledImage(image=reconstructedInput,
                           path=self._layers[layerIndex].layerID + '_reconstruction_' + 
                                str(globalEpoch+localEpoch) + '.png',
                           imageShape=tuple(self._layers[layerIndex].getInputSize()[-2:]),
                           spacing=1,
                           interleave=True)
            '''

            locCost = np.mean(locCost, axis=0)
            self._startProfile(layerEpochStr + ' Cost: ' + \
                               str(locCost[0]) + ' - Jacob: ' + \
                               str(locCost[1]) + ' - Sparsity: ' + \
                               str(locCost[2]), 'info')
            globCost.append(locCost)

            self._endProfile()
            self._endProfile()

            #self.writeWeights(layerIndex, globalEpoch + localEpoch)
        return globalEpoch + numEpochs, globCost


