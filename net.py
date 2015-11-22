from layer import Layer
from profiler import Profiler
import theano
import theano.tensor as t

class Net () :
    ''' '''
    def __init__ (self, learningRate, runCPU=True, log=None) :
#        self._patterSize = patternSize
        self._runCPU = runCPU
        self._profiler = Profiler(log, 
                                  'NeuralNet', 
                                  './ApplicationName-Profile.xml') if \
                                  log is not None else None
        self._learningRate = learningRate
        self._layers = []
        self._weights = []
        self.input = None
        self.output = None

    def _listify(self, data) :
        if data is None : return []
        else : return data if isinstance(data, list) else [data]

    def getNetworkInput(self) :
        '''Return the first layer's input'''
        if len(self._layers) == 0 :
            raise IndexError('Network must have at least one layer' +
                             'to call getNetworkInput().')
        return self._layers[0].input

    def getNetworkOutput(self) :
        '''Return the last layer's output. This should be used as input to
           the next layer.
        '''
        if len(self._layers) == 0 :
            raise IndexError('Network must have at least one layer' +
                             'to call getNetworkOutput().')
        return self._layers[-1].output

    def addLayer(self, layer) :
        '''Add a Layer to the network. It is the responsibility of the user
           to connect the current network's output as the input to the next
           layer.
        '''
        if not isinstance(layer, Layer) :
            raise TypeError('addLayer is expecting a Layer object.')
        if self._profiler is not None :
            self._profiler.startProfile('Adding a layer to the network',
                                        'debug')
        # add it to our layer list
        self._layers.append(layer)
        
        # weight/bias are added in reverse order because they will
        # be used back propagation, which runs output to input
        self._weights.insert(0, layer.getWeights())

    def finalizeNetwork(self) :
        '''Setup the '''
        if self._profiler is not None :
            self._profiler.startProfile('Finalizing Network', 'info')

        # create one function that activates the entire network
        out = self._layers[-1].output
        self._classify = theano.function([self._layers[0].input], out)

        # create a softmax function -- 
        # this is the normalized output prediction, which emphasizes
        # significant neural responses. This takes as input the output
        # of the final layer (ie the network classification).
        sm = t.nnet.softmax(out)
        self._softmax = theano.function([out], sm)

        # create the negative log likelihood function --
        # the network uses this as a cost function to optimize
        expectedOutput = t.fvector('expectedOutput')
        cost = -t.mean(t.log(sm)[t.arange(sm.shape[0]), expectedOutput])

        # create the function for back propagation of all layers
        # this is combined for convenience
        gradients = t.grad(cost, self._weights)
        updates = [(weights, weights - self._learningRate * gradient)
                   for weights, gradient in zip(self._weights, gradients)]
        self._trainNetwork = theano.function(
            [self._layers[0].input, expectedOutput], cost, updates=updates)


    def classify (self, input) :
        '''Classify the given input'''
        if self._profiler is not None :
            self._profiler.startProfile('Classifying the Input', 'info')
        if not hasattr(self, '_classify') :
            self.finalizeNetwork()

        # activating the last layer triggers all previous 
        # layers due to dependencies we've enforced
        out = self._classify(input)

        # use softmax as the classifier --
        # argmax returns the index of the max classification
        pred = t.argmax(self._softmax(out), axis=1)
        self._profiler.endProfile()

        # return both in case we want to perform different cost functions
        return (pred, out)
        
    def train(self, input, expectedOutput) :
        '''Train the network against this input. This accepts single input or
           full lists of inputs.
        '''
        if self._profiler is not None :
            self._profiler.startProfile('Classifying the Input', 'info')
        # this uses online training
        for i in self._listify(input) :
            self._trainNetwork(i)
            
        

if __name__ == "__main__" :
    '''    
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

    input = t.fscalar('input')
    expectedOutput = t.fscalar('expectedOutput')

    network = Net(.001, True)
    network.addLayer(ContiguousLayer('c1', input, 5, 3))
    network.addLayer(ContiguousLayer('c2', network.getNetworkOutput(), 3, 3))
    (pred, out) = network.classify([1, 2, 3, 4, 5])
    print (pred, out)
