from theano.tensor.shared_randomstreams import RandomStreams
from time import time
from nn.opMap import convertActivation

class Layer () :
    # rng is across all layer types
    _randStream = RandomStreams(int(time()))

    def __init__ (self, layerID, learningRate=0.001, 
                  momentumRate=0.9, dropout=None, activation=None) :
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
        self._weights = None
        self._thresholds = None
        self._learningRate = learningRate
        self._momentumRate = momentumRate
        self._dropout = dropout
        self._activation = activation

    def __getstate__(self) :
        '''Save layer pickle'''
        from dataset.shared import fromShared
        dict = self.__dict__.copy()
        dict['input'] = None
        dict['output'] = None
        dict['_weights'] = fromShared(self._weights)
        dict['_thresholds'] = fromShared(self._thresholds)
        # convert to a string for pickling purposes
        dict['_activation'] = convertActivation(self._activation)
        return dict

    def __setstate__(self, dict) :
        '''Load layer pickle'''
        from theano import shared
        self.__dict__.update(dict)
        initialWeights = self._weights
        self._weights = shared(value=initialWeights, borrow=True)
        initialThresholds = self._thresholds
        self._thresholds = shared(value=initialThresholds, borrow=True)
        # convert back to a theano operation
        self._activation = convertActivation(self._activation)

    def _setActivation(self, out) :
        return out if self._activation is None else self._activation(out)

    def _setOutput(self, outSize, outClass, outTrain) :
        from theano.tensor import switch

        # determine dropout if requested
        if self._dropout is not None :
            # here there are two possible paths --
            # outClass : path of execution intended for classification. Here
            #            all neurons are present and weights must be scaled by
            #            the dropout factor. This ensures resultant 
            #            probabilities fall within intended bounds when all
            #            neurons are present.
            # outTrain : path of execution for training with dropout. Here each
            #            neuron's output goes through a Bernoulli Trial. This
            #            retains a neuron with the probability specified by the
            #            dropout factor.
            outClass = outClass / self._dropout
            outTrain = switch(self._randStream.binomial(
                size=outSize, p=self._dropout), outTrain, 0)

        # activate the layer --
        # output is a tuple to represent two possible paths through the
        # computation graph.
        return (self._setActivation(outClass), self._setActivation(outTrain))

    def finalize(self) :
        raise NotImplementedError('Implement the finalize() method')

    def getWeights(self) :
        '''This allows the network backprop all layers efficiently.'''
        return [self._weights, self._thresholds]

    def getMomentumRate(self) :
        return self._momentumRate

    def getLearningRate(self) :
        return self._learningRate

    def getInputSize (self) :
        raise NotImplementedError('Implement the getInputSize() method')

    def getOutputSize (self) :
        raise NotImplementedError('Implement the getOutputSize() method')
