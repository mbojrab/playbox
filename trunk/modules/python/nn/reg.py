class Regularization :
    '''Object to save regularization constraints inside a network.

       regType     : Type of regularization ('L1' or 'L2')
       scaleFactor : Scale factor to apply on the regularization
    '''
    def __init__(self, regType, scaleFactor=1.) :
        self._type = regType
        self._scale = scaleFactor / 2. if regType == 'L2' else scaleFactor

    def calculate(self, layers) :
        '''Calculate the type of regularization given weights from the network.

           layers : Layers of a network
        '''
        from nn.costUtils import leastAbsoluteDeviation, leastSquares
    
        # calculate a regularization term -- if desired
        reg = None

        # L1-norm provides 'Least Absolute Deviation' --
        # built for sparse outputs and is resistent to outliers
        if self._type == 'L1' :
            reg = leastAbsoluteDeviation(
                [layer.getWeights()[0] for layer in layers],
                batchSize=None, scaleFactor=self._scale)
    
        # L2-norm provides 'Least Squares' --
        # built for dense outputs and is computationally stable at small errors
        elif self._type == 'L2' :
            reg = leastSquares(
                [layer.getWeights()[0] for layer in layers],
                batchSize=None, scaleFactor=self._scale)
    
        return reg
