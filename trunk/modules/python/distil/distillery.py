class Distillery :
    '''This object creates dataset pickles capable of distilling a 
       deep network into a shallower network. The goal of this is to create
       a network equally capable of classification, yet faster to operate.
    '''
    def __init__ (self, network, pickleFile) :
        self._net = network
        self._pickle = pickleFile
    def convertDataset
    
    