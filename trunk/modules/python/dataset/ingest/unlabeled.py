def ingestImagery(filepath, shared=False, log=None, *func, **kwargs) :
    '''Load the unlabeled dataset into memory. This reads and chips any
       imagery found within the filepath according the the options sent to the
       function.

       filepath : This can be a cPickle, a path to the directory structure.
       shared   : Load data into shared variables for training
       log      : Logger for tracking the progress
       func     : Chipping utility to use on each image
       kwargs   : Parameters 
       return   : (trainingData, pixelRegion=None)
    '''
    raise Exception('Implement ingest.unlabeled.ingestImagery()')