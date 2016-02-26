import theano.tensor as t
import theano, cPickle, gzip
from nn.net import Network
from encoder import AutoEncoder

class StackedAENetwork (Network) :
    '''The StackedAENetwork object allows autoencoders to be stacked such that
       the output of one autoencoder becomes the input to another. It creates
       the necessary connections to train the AE in a greedy layerwise manner. 
       The resulting trained AEs can be used to initialize a nn.TrainerNetwork.

       train : theano.shared dataset used for network training in format --
               (numBatches, batchSize, numChannels, rows, cols)
       log   : Logger to use
    '''
    def __init__ (self, train, log=None) :
        Network.__init__ (self, log)
        self._trainData, self._trainLabels = train
        self._greedyTrainer = []

    def addLayer(self, encoder) :
        '''Add an autoencoder to the network. It is the responsibility of the 
           user to connect the current network's output as the input to the 
           next layer.
           This utility will additionally create a greedy layerwise trainer.
        '''
        if not isinstance(encoder, AutoEncoder) :
            raise TypeError('addLayer is expecting a AutoEncoder object.')
        self._startProfile('Adding a Encoder to the network', 'debug')

        # add it to our layer list
        self._layers.append(encoder)

        # all layers start with the input original input, however are updated
        # in a layerwise manner.
        out, update = encoder.getUpdates()
        self._greedyTrainer.append(
            theano.function(self.getNetworkInput(), out, updates=update))
        self._endProfile()

    def train(self, layerIndex, index, inputs) :
        '''Train the network against the pre-loaded inputs. This accepts
           a batch index into the pre-compiled input set.
           layerIndex : specify which layer to train
           index      : specify a pre-compiled mini-batch index
           inputs     : DEBUGGING Specify a numpy tensor mini-batch
        '''
        self._startProfile('Training Batch [' + str(index) +
                           '/' + str(self._numTrainBatches) + ']', 'debug')
        if not isinstance(index, int) :
            raise Exception('Variable index must be an integer value')
        if index >= self._numTrainBatches :
            raise Exception('Variable index out of range for numBatches')

        # train the input --
        # the user decides if this is online or batch training
        self._greedyTrainer[layerIndex](inputs)
        #self._greedyTrainer[layerIndex](index)

        self._endProfile()
    def trainEpoch(self, layerIndex, globalEpoch, numEpochs=1) :
        '''Train the network against the pre-loaded inputs for a user-specified
           number of epochs.
           globalEpoch : total number of epochs the network has previously 
                         trained
           numEpochs   : number of epochs to train this round before stopping
        '''
        for localEpoch in range(numEpochs) :
            self._startProfile('Running Epoch [' + 
                               str(globalEpoch + localEpoch) + ']', 'info')
            for ii in range(self._numTrainBatches) :
                self.train(layerIndex, ii, self._trainData[ii])
            self._endProfile()
        return globalEpoch + numEpochs
