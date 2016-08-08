import os

def readAndDivideData(path, holdoutPercentage, minTest, log=None) :
    '''This walks the directory structure and divides the data according to the
       user specified holdout over two "train" and "test" sets.
    '''
    from dataset.reader import readImage
    train, test, labels = [], [], []
    for root, dirs, files in os.walk(path) :
        if root == path :
            continue
        if len(files) == 0 :
            if log is not None :
                log.debug('No files found in [' + root + ']')
            continue

        # add the new label for the current directory
        label = os.path.relpath(root, path).replace(os.path.sep, '.')
        labels.append(label)
        indx = len(labels) - 1
        if log is not None :
            log.debug('Adding directory [' + root + '] as [' + label + ']')

        # read the imagery and assign it this label --
        # a small percentage of the data is held out to verify our training
        # isn't getting overfitted. We will randomize the input later.
        numTest = max(minTest, int(holdoutPercentage * len(files)))
        if log is not None :
            log.debug('Holding out [' + str(numTest) + '] of [' + \
                      str(len(files)) + ']')

        # this is updated to at least holdout a few images for testing
        holdoutPercentage = (float(numTest) / float(len(files)))

        holdoutPerc = holdoutPercentage
        if holdoutPercentage > .5 :
            holdoutPerc = 1. - holdoutPerc

        holdout = len(files) if holdoutPerc == 0. else \
                      int(round(1. / holdoutPerc))

        # this loop ensures a good sampling is held out across our entire
        # dataset. This is just in case there is some regularity in the data
        # as it sits on disk.
        te, tr = [], []
        for ii, file in enumerate(files) :
            try :
                imgLabel = readImage(os.path.join(root, file), log), indx
                te.append(imgLabel) if (ii+1) % holdout == 0 else \
                    tr.append(imgLabel)
            except IOError :
                # continue on if image cannot be read
                pass

        # swap these buffers to align with what the user specified
        if holdoutPercentage > .5 :
            (te, tr) = (tr, te)

        # populate the global buffers
        train.extend(tr)
        test.extend(te)

    return train, test, labels

def pickleDataset(filepath, holdoutPercentage=.05, minTest=5,
                  batchSize=1, log=None) :
    '''Create a pickle out of a directory structure. The directory structure
       is assumed to be a series of directories, each contains imagery assigned
       the label of the directory name.

       filepath          : Top-level directory containing the label directories
       holdoutPercentage : Percentage of the data to holdout for testing
       minTest           : Hard minimum on holdout if percentage is low
       batchSize         : Size of a mini-batch
       log               : Logger to use
    '''
    from dataset.minibatch import makeContiguous
    from dataset.shuffle import naiveShuffle
    from dataset.pickle import writePickleZip

    rootpath = os.path.abspath(filepath)
    outputFile = os.path.join(rootpath, os.path.basename(rootpath) + 
                              '_labeled' + 
                              '_holdout_' + str(holdoutPercentage) +
                              '_batch_' + str(batchSize) +'.pkl.gz')
    if os.path.exists(outputFile) :
        if log is not None :
            log.info('Pickle exists for this dataset [' + outputFile +
                     ']. Using this instead.')
        return outputFile

    # walk the directory structure
    if log is not None :
        log.info('Reading the directory structure')

    # we support two possible operations --
    # If there are train/ and test/ directories in the target directories, we
    # use the user provided breakdown of the data and ignore the holdout
    # parameter. Otherwise, we walk the unified directory structure and divide
    # the data according to the holdoutPercentage.
    trainDir = os.path.join(rootpath, 'train')
    testDir = os.path.join(rootpath, 'test')
    if os.path.isdir(trainDir) and os.path.isdir(testDir) :
        
        # run each directory to grab the imagery
        trainSet = readAndDivideData(trainDir, 0., 0, log)
        testSet = readAndDivideData(testDir, 1., 0, log)

        # verify the labels overlap
        for label in testSet[-1] :
            if label not in trainSet[-1] :
                raise ValueError('Train and Test sets have non-overlapping ' +
                                 'labels. Please check the input directory.')

        # setup for further processing
        train, test, labels = trainSet[0], testSet[1], trainSet[-1]

    else :
        # split the data according to the user-provided holdout
        train, test, labels = readAndDivideData(rootpath, holdoutPercentage,
                                                minTest, log)

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

def ingestImagery(filepath, shared=True, log=None, **kwargs) :
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

    if not os.path.exists(filepath) :
        raise ValueError('The path specified does not exist.')

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
