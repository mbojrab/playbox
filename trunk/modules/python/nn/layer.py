from theano.tensor.shared_randomstreams import RandomStreams
from time import time

class Layer () :
    # rng is across all layer types
    _randStream = RandomStreams(int(time()))

    def __init__ (self, layerID, learningRate=0.001, 
                  momentumRate=0.9, dropout=None) :
        '''This class describes an abstract Neural Layer.
           layerID      : unique name identifier for this layer
           learningRate : multiplier for speed of gradient descent 
           momentumRate : multiplier for force of gradient descent
           dropout      : rate of retention in a given neuron during training
        '''
        # input can be a tuple or a variable
        self.input = None
        # output must be a tuple
        self.output = None
        self.layerID = layerID
        self._learningRate = learningRate
        self._momentumRate = momentumRate
        self._dropout = dropout

    def getWeights(self) :
        raise NotImplementedError('Implement the getWeights() method')

    def getMomentumRate(self) :
        return self._momentumRate

    def getLearningRate(self) :
        return self._learningRate

    def getInputSize (self) :
        raise NotImplementedError('Implement the getInputSize() method')

    def getOutputSize (self) :
        raise NotImplementedError('Implement the getOutputSize() method')
