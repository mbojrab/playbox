def ingestImagery(filepath, shared=False, log=None, **kwargs) :
    '''Load the labeled dataset into memory. This is formatted such that the
       directory structure becomes the labels, and all imagery within the 
       directory will be assigned this label. All images in any directory is
       required to have the same dimensions.

       filepath : This can be a cPickle, a path to the directory structure.
       shared   : Load data into shared variables for training
       log      : Logger for tracking the progress
       kwargs   : Any parameters needed to override defaults in pickleDataset
       return   :
           Format -- 
           (trainData, trainLabel), (testData, testLabel), labels

           The 'trainLabel' and 'testLabel' are integer values corresponding
           to the index into the 'labels' string vector. this provides a
           better means to identify errors during back propagation, but
           still allows label finding during classification.

           TODO: Consider returning these as objects for more intuitive
                 indexing. For now numpy indexing is sufficient.
    '''
    import os
    from dataset.shared import splitToShared
    from dataset.pickle import readPickleZip
    from dataset.reader import pickleDataset

    # read the directory structure and pickle it up
    if os.path.isdir(filepath) :
        filepath = pickleDataset(filepath, log=log, **kwargs)

    # Load the dataset to memory
    train, test, labels = readPickleZip(filepath, log)

    # load each into shared variables -- 
    # this avoids having to copy the data to the GPU between each call
    if shared is True :
        if log is not None :
            log.debug('Transfer the memory into shared variables')
        return splitToShared(train), splitToShared(test), labels
    else :
        return train, test, labels
