import os
import numpy as np

def readAndDivideData(path, holdoutPercentage, minTest=5, log=None) :
    '''This walks the directory structure and divides the data according to the
       user specified holdout over two "train" and "test" sets.
    '''
    from dataset.shuffle import naiveShuffle

    # read the directory structure --
    # each subdirectory becomes a label and the imagery within are examples.
    # Splitting the data per label ensures each category is represented in the
    # holdout set.
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

        # a small percentage of the data is held out to verify our training
        # isn't getting overfitted. We will randomize the input later.
        numTest = max(minTest, int(holdoutPercentage * len(files)))
        if log is not None :
            log.debug('Holding out [' + str(numTest) + '] of [' + \
                      str(len(files)) + ']')

        # this ensures a minimum number of examples are used for testing
        holdoutPercentage = (float(numTest) / float(len(files)))

        # prepare the data
        items = np.asarray(
            [(os.path.join(root, file), indx) for file in files],
            dtype=np.object)
        naiveShuffle(items)

        # randomly distribute the data using Random Assignment based on
        # Bernoulli trials
        # TODO: There may be a more compact way to represent this in python
        randomAssign = np.random.binomial(1, holdoutPercentage, len(items))
        train.extend(
            [items[ii] for ii in range(len(items)) if randomAssign[ii] == 0])
        test.extend(
            [items[ii] for ii in range(len(items)) if randomAssign[ii] == 1])

    # randomize the data across categories -- otherwise its not stochastic
    if log is not None :
        log.info('Shuffling the data for randomization')
    naiveShuffle(train)
    naiveShuffle(test)

    return train, test, labels

def hdf5Dataset(filepath, holdoutPercentage=.05, minTest=5,
                batchSize=1, log=None) :
    '''Create a hdf5 file out of a directory structure. The directory structure
       is assumed to be a series of directories, each contains imagery assigned
       the label of the directory name.

       filepath          : Top-level directory containing the label directories
       holdoutPercentage : Percentage of the data to holdout for testing
       minTest           : Hard minimum on holdout if percentage is low
       batchSize         : Size of a mini-batch
       log               : Logger to use
    '''
    import theano
    from dataset.reader import readImage, getImageDims
    from dataset.hdf5 import createHDF5Labeled

    rootpath = os.path.abspath(filepath)
    outputFile = os.path.join(rootpath, os.path.basename(rootpath) + 
                              '_labeled' + 
                              '_holdout_' + str(holdoutPercentage) +
                              '_batch_' + str(batchSize) +'.hdf5')
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

    if len(train) == 0 :
        raise ValueError('No training examples found [' + filepath + ']')

    # create a hdf5 memmap --
    # Here we create the handles to the data buffers. This operates on the
    # assumption the dataset may not fit entirely in memory. The handles allow
    # data to overflow and ultimately be stored entirely on disk. 
    imageShape = list(getImageDims(train[0][0], log))
    trainShape = [len(train) // batchSize, batchSize] + imageShape
    testShape = [len(test) // batchSize, batchSize] + imageShape

    if log is not None :
        log.info('Creating the memmapped HDF5')

    [handleH5, trainDataH5, trainIndicesH5, 
     testDataH5, testIndicesH5, labelsH5] = \
        createHDF5Labeled (outputFile, 
                           trainShape, theano.config.floatX, np.int32,
                           testShape, theano.config.floatX, np.int32,
                           len(labels), log)

    if log is not None :
        log.info('Writing data to archive')

    # populate the indice buffers
    trainIndicesH5[:] = np.resize(np.asarray(train).flatten()[1::2],
                                  trainIndicesH5.shape).astype(np.int32)[:]
    testIndicesH5[:] = np.resize(np.asarray(test).flatten()[1::2],
                                 testIndicesH5.shape).astype(np.int32)[:]

    # stream the imagery into the buffers --
    # NOTE : h5py.Dataset doesn't implement __setslice__, so we must implement
    #        the copy via __setitem__. This differs from my normal index
    #        formatting, but it gets the job done.
    for ii in range(np.prod(trainShape[:1])) :
        trainDataH5[ii // batchSize, ii % batchSize, :] = \
              readImage(train[ii][0], log)
    for ii in range(np.prod(testShape[:1])) :
        testDataH5[ii // batchSize, ii % batchSize, :] = \
              readImage(test[ii][0], log)

    # stream in the label in string form
    labelsH5[:] = labels[:]

    if log is not None :
        log.info('Flushing to disk')

    # write it to disk    
    handleH5.flush()
    handleH5.close()

    # return the output filename
    return outputFile

def ingestImagery(filepath, shared=True, log=None, **kwargs) :
    '''Load the labeled dataset into memory. This is formatted such that the
       directory structure becomes the labels, and all imagery within the 
       directory will be assigned this label. All images in any directory is
       required to have the same dimensions.

       filepath : This can be a cPickle, a path to the directory structure.
       shared   : Load data into shared variables for training --
                  NOTE: this is only a user suggestion. However the size of the
                        data will ultimately determine how its loaded.
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
    import psutil
    import theano.tensor as t
    from dataset.shared import splitToShared
    from dataset.hdf5 import readHDF5

    if not os.path.exists(filepath) :
        raise ValueError('The path specified does not exist.')

    # read the directory structure and pickle it up
    if os.path.isdir(filepath) :
        filepath = hdf5Dataset(filepath, log=log, **kwargs)

    # Load the dataset to memory
    train, test, labels = readHDF5(filepath, log)

    # calculate the memory needed by this dataset
    if t.config.floatX == 'float32' :
        dataMemoryConsumption = (np.prod(train[0].shape) + 
                                 np.prod(train[1].shape) + 
                                 np.prod(test[0].shape) + 
                                 np.prod(test[1].shape)) * 4
    else :
        dataMemoryConsumption = np.prod(train[0].shape) * 8 + \
                                np.prod(train[1].shape) * 4 + \
                                np.prod(test[0].shape)  * 8 + \
                                np.prod(test[1].shape)  * 4
    memoryConsumGB = str(dataMemoryConsumption / 1024. / 1024. / 1024.)

    # check if the machine is capable of loading dataset into CPU memory
    oneGigMem = 2 ** 30
    availableCPUMem = psutil.virtual_memory()[1] - oneGigMem
    if availableCPUMem > dataMemoryConsumption :
        if log is not None :
            log.info('Dataset loaded into CPU memory. [' + 
                      memoryConsumGB + '] GBs consumed.')
    else :
        if log is not None :
            log.info('Dataset is too large for CPU memory. Dataset will ' +
                      'be memory mapped and backed by disk IO. [' + 
                      memoryConsumGB + '] GBs consumed.')

    # if the user wants to use the GPU check if the dataset can be loaded 
    # entirely into shared memory
    if 'gpu' in t.config.device and shared :
        import theano.sandbox.cuda.basic_ops as sbcuda

        # the user has requested this goes into GPU memory if possible
        availableGPUMem = sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]
        if availableGPUMem > dataMemoryConsumption :
            if log is not None :
                log.info('Loading dataset into GPU memory. [' + 
                memoryConsumGB + '] GBs consumed.')
        else :
            if log is not None :
                log.info('Dataset is too large for GPU memory. ' + 
                'Dataset is [' + memoryConsumGB + '] GBs.')
            shared = False

    # load each into shared variables -- 
    # this avoids having to copy the data to the GPU between each call
    if shared is True :
        if log is not None :
            log.debug('Transfer the memory into shared variables')
        return splitToShared(train), splitToShared(test), labels
    else :
        return train, test, labels
