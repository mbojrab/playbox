
def pickleDataset(filepath, holdoutPercentage=.05, minTest=5,
                  batchSize=1, log=None) :
    '''Create a pickle out of a directory structure. The directory structure
       is assumed to be a series of directories, each contains imagery assigned
       the label of the directory name.

       filepath : path to the top-level directory contain the label directories
    '''
    import os
    from dataset.minibatch import makeContiguous
    from dataset.shuffle import naiveShuffle
    from dataset.pickle import writePickleZip
    from dataset.reader import readImage

    rootpath = os.path.abspath(filepath)
    outputFile = os.path.join(rootpath, os.path.basename(rootpath) + 
                              '_holdout_' + str(holdoutPercentage) +
                              '_batch_' + str(batchSize) +'.pkl.gz')
    if os.path.exists(outputFile) :
        if log is not None :
            log.info('Pickle exists for this dataset [' + outputFile +
                     ']. Exiting')
        return outputFile

    # walk the directory structure
    if log is not None :
        log.info('Reading the directory structure')
    train, test, labels = [], [], []
    for root, dirs, files in os.walk(rootpath) :
        if root == rootpath :
            continue
        if len(files) == 0 :
            if log is not None :
                log.debug('No files found in [' + root + ']')
            continue

        # add the new label for the current directory
        label = os.path.relpath(root, rootpath).replace(os.path.sep, '.')
        labels.append(label)
        indx = len(labels) - 1
        if log is not None :
            log.debug('Adding directory [' + root + '] as [' + label + ']')

        # read the imagery and assign it this label --
        # a small percentage of the data is held out to verify our training
        # isn't getting overfitted. We will randomize the input later.
        numTest = max(minTest, int(holdoutPercentage * len(files)))

        # this is updated to at least holdout a few images for testing
        holdoutPercentage = (float(numTest) / float(len(files)))
        holdoutTest  = int(1. / holdoutPercentage)
        holdoutTrain = int(1. / (1.-holdoutPercentage))
        if log is not None :
            log.debug('Holding out [' + str(numTest) + '] of [' + \
                      str(len(files)) + ']')

        # this loop ensures a good sampling is held out across our entire
        # dataset. if holdoutPercentage is <.5 this takes the if, otherwise
        # uses the else.
        for ii in range(len(files)) :
            try :
                imgLabel = readImage(os.path.join(
                                root, files[ii]), log), indx
                if holdoutTest > 1 :
                    test.append(imgLabel) if ii % holdoutTest == 0 else \
                        train.append(imgLabel)
                else :
                    train.append(imgLabel) if ii % holdoutTrain == 0 else \
                        test.append(imgLabel)
            except IOError :
                # continue on if image cannot be read
                pass

    # randomize the data -- otherwise its not stochastic
    if log is not None :
        log.info('Shuffling the data for randomization')
    naiveShuffle(train)
    naiveShuffle(test)

    # create mini-batches
    if log is not None :
        log.info('Creating the mini-batches')
    train = makeContiguous(train, batchSize)
    test =  makeContiguous(test, batchSize)

    # pickle the dataset
    writePickleZip(outputFile, (train, test, labels), log)

    # return the output filename
    return outputFile

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
