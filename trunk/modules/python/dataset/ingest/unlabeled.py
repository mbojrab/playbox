def pickleDataset(filepath, holdoutPercentage=.05, minTest=5,
                  batchSize=1, log=None, *chipFunc, **kwargs) :
    '''Create a pickle out of a directory structure. The directory structure
       is assumed to be a series of directories, each contains imagery assigned
       the label of the directory name.

       filepath          : path to the top-level directory contain the images
       holdoutPercentage : Percentage of the data to holdout for testing
       minTest           : Hard minimum on holdout if percentage is low
       batchSize         : Size of a mini-batch
       log               : Logger to use
       *chipFunc         : Chipping function to use
       **kwargs          : Chipping function parameters
    '''
    import os
    from dataset.minibatch import makeContiguous
    from dataset.shuffle import naiveShuffle
    from dataset.pickle import writePickleZip
    from dataset.reader import readImage
    from dataset.chip import prepareChips

    rootpath = os.path.abspath(filepath)
    outputFile = os.path.join(rootpath, os.path.basename(rootpath) +
                              '_unlabeled' + 
                              '_chip_' + chipFunc.__name__ + 
                              '_holdout_' + str(holdoutPercentage) + 
                              '_batch_' + str(batchSize) + '.pkl.gz')
    if os.path.exists(outputFile) :
        if log is not None :
            log.info('Pickle exists for this dataset [' + outputFile +
                     ']. Exiting')
        return outputFile

    # chip the imagery one at a time to save on memory
    if log is not None :
        log.info('Reading the Imagery')
    chips = []
    for img in os.listdir(rootpath) :
        try :
            chips.extend(chipFunc(readImage(
                os.path.join(rootpath, img), log), **kwargs))
        except IOError : pass

    # randomize the data -- otherwise its not stochastic
    if log is not None :
        log.info('Shuffling the data for randomization')
    naiveShuffle(chips)

    # create mini-batches
    if log is not None :
        log.info('Creating the mini-batches')
    chips, regions = prepareChips(chips=chips, pixelRegion=False,
                                  batchSize=batchSize, log=None)

    # TODO: still needs division over two sets

    # pickle the dataset
    writePickleZip(outputFile, chips, log)

    # return the output filename
    return outputFile

def ingestImagery(filepath, shared=False, log=None, *chipFunc, **kwargs) :
    '''Load the unlabeled dataset into memory. This reads and chips any
       imagery found within the filepath according the the options sent to the
       function.

       filepath : This can be a cPickle, a path to the directory structure.
       shared   : Load data into shared variables for training
       log      : Logger for tracking the progress
       chipFunc : Chipping utility to use on each image
       kwargs   : Parameters specific for the chipping function
       return   : (trainingData, pixelRegion={None})
    '''
    import os
    from dataset.shared import splitToShared
    from dataset.pickle import readPickleZip
    from dataset.reader import pickleDataset

    # read the directory structure and pickle it up
    images = []
    if not isinstance(filepath, list) :
        filepath = [filepath]
    while filepath :
        im = filepath.pop(0)
        if os.path.isdir(im):
            images.extend(map(lambda x: os.path.join(im, x), os.listdir(im)))
        else :
            images.append(im)

    if os.path.isdir(filepath) :
        filepath = pickleDataset(filepath, log=log, *chipFunc, **kwargs)

    # Load the dataset to memory
    train, test = readPickleZip(filepath, log)

    # load each into shared variables -- 
    # this avoids having to copy the data to the GPU between each call
    if shared is True :
        if log is not None :
            log.debug('Transfer the memory into shared variables')
        return splitToShared(train), splitToShared(test)
    else :
        return train, test
