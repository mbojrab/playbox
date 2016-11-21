class AutoEncoder () :
    def __init__ (self, contractionRate=0.01) :
        '''This class describes an abstract AutoEncoder.
           contractionRate  : variance (dimensionality) reduction rate
        '''
        self._contractionRate = contractionRate

    def buildDecoder(self, input) :
        raise NotImplementedError('Implement the buildDecoder() method')

    def getUpdates(self) :
        raise NotImplementedError('Implement the getUpdates() method')

    # DEBUG: For Debugging purposes only
    def saveReconstruction(self, image, ii) :
        from dataset.debugger import saveNormalizedImage
        saveNormalizedImage(np.resize(self.reconstruction(image), 
                                      image.shape[-2:]),
                            'chip_' + str(ii) + '_reconst.png')
