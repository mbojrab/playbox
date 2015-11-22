class Layer () :
    def __init__ (self, layerID, runCPU=True) :
        '''This class describes an abstract Neural Layer.
           layerID : unique name identifier for this layer
           runCPU  : run processing on CPU
        '''
        self.input = None
        self.output = None
        self.layerID = layerID
        self.runCPU = runCPU
        
    def getWeights(self) :
        raise NotImplementedError('Implement the getWeights() method')

    def activate (self, input) :
        raise NotImplementedError('Implement the activate() method')

    def backPropagate (self, input, errorGrad, backError) :
        raise NotImplementedError('Implement the backPropagate() method')

    def getInputSize (self) :
        raise NotImplementedError('Implement the getInputSize() method')

    def getOutputSize (self) :
        raise NotImplementedError('Implement the getOutputSize() method')
