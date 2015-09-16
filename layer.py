from exception import NotImplementedError

class Layer () :
    def __init__ (self, layerID, runCPU=True) :
        '''This class describes an abstract Neural Layer.
           layerID : unique name identifier for this layer
           runCPU  : run processing on CPU
        '''
        self.layerID = layerID
        self.runCPU = runCPU

    def activate (self, input) :
        raise NotImplementedError('Implement the activate() method')

    def backPropagate (self, input, errorGrad, backError) :
        raise NotImplementedError('Implement the backPropagate() method')

    def getInputSize (self) :
        raise NotImplementedError('Implement the getInputSize() method')

    def getOutputSize (self) :
        raise NotImplementedError('Implement the getOutputSize() method')
