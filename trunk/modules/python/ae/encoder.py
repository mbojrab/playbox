class AutoEncoder () :
    def __init__ (self, contractionRate=0.01) :
        '''This class describes an abstract AutoEncoder.
           contractionRate  : variance (dimensionality) reduction rate
        '''
        self._contractionRate = contractionRate

    def getUpdates(self) :
        raise NotImplementedError('Implement the getUpdates() method')
