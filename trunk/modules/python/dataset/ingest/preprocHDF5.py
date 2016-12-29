import os
import numpy as np
import theano.tensor as t

def checkAvailableMemory(dataMemoryConsumption, shared, log) :
    '''There are three possible cases of memory consumption:
       1. GPU has enough memory, thus load the dataset directly to the device.
          This will achieve the best performance as transfers to the GPU over
          PCIe will starve its processing.
       2. GPU is insufficient on memory, but it can fit in CPU memory. We
          load the entire dataset in CPU memory and transfer over PCIe
          just-in-time for processing.
       3. CPU and GPU memory is insufficient. In this case we rely on disk IO
          to support training on this large dataset. Performance will be
          extremely dampened while running in this mode, however training will
          be possible.

       NOTE: Cases 2 & 3 are hidden from the user by using HDF5. No additional
             care will need to be take between these two processing types.
    '''
    import psutil

    convertToGB = 1. / (1024. * 1024. * 1024.)
    memoryConsumGB = str(dataMemoryConsumption * convertToGB)
    if log is not None :
        log.info('Dataset will consume [' + memoryConsumGB + '] GBs')

    # check if the machine is capable of loading dataset into CPU memory --
    # NOTE: this assumes there is more CPU memory (RAM) than GPU memory (DRAM)
    availableCPUMem = psutil.virtual_memory()[1]
    if availableCPUMem > dataMemoryConsumption :
        if log is not None :
            log.debug('Dataset will fit in CPU memory. [' + 
                      str(availableCPUMem * convertToGB) + '] GBs available.')
    else :
        if log is not None :
            log.warn('Dataset is too large for CPU memory. Dataset will ' +
                     'be memory mapped and backed by disk IO.')
        shared = False

    # if the user wants to use the GPU check if the dataset can be loaded 
    # entirely into shared memory
    if 'gpu' in t.config.device and shared :
        import theano.sandbox.cuda.basic_ops as sbcuda

        # the user has requested this goes into GPU memory if possible --
        # NOTE: this check is by no means guaranteed. There must be enough
        #       contigous memory on the device for a successful allocation.
        #       Below we handle the case where this check passes, but
        #       the allocation ultimately fails.
        availableGPUMem = sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]
        if availableGPUMem > dataMemoryConsumption :
            if log is not None :
                log.debug('Dataset will fit in GPU memory. [' + 
                          str(availableGPUMem * convertToGB) + 
                          '] GBs available.')
        else :
            if log is not None :
                log.warn('Dataset is too large for GPU memory. Dataset will ' + 
                         'be transferred over PCIe just-in-time. ')
            shared = False

    return shared

def preprocessData(rootpath, image, log=None) :
    '''This function reads and preprocesses images residing in common
       image formats or as datasets within HDF5.

       NOTE: This handles both ingest from a file on disk or within
             an HDF5 file representing the filesystem.

       rootpath : Path to HDF5 file or parent directory
       image    : Path to image. This is absolute path for imagery on
                  disk, and relative path when using HDF5.
       log      : Logger to use
       
    '''
    from dataset.reader import preProcImage
    if not os.path.isdir(rootpath) :
        import h5py
        with h5py.File(rootpath, 'r') as f :
            group = f
            for ii in image.split(os.sep) :
                image = group = group[ii]
            image = image[:]
    return preProcImage(image, log)

def populateDataHDF5(h5Dataset, rootpath, data, dataShape,
                     batchSize, threads, log) :
    '''Read and preprocess the data for use in network training.
       This streams the data into the appropriate location of an
       HDF5 file, such that it can be reused later.

       h5Dataset : HDF5 file handle to populate
       rootpath  : Parent directory or HDF5 where the files reside
       data      : List of all imagery paths. These are already sorted.
       dataShape : All data should be this size, and pad if its smaller
       batchSize : The size of the batch for the imagery
       threads   : Number of python threads to use
       log       : Logger to use
    '''
    import theano
    from six.moves import queue
    import threading
    from dataset.reader import padImageData, preProcImage

    # add jobs to the queue --
    # NOTE : h5py.Dataset doesn't implement __setslice__, so we must implement
    #        the copy via __setitem__. This differs from my normal index
    #        formatting, but it gets the job done.
    workQueueData = queue.Queue()
    for ii in range(dataShape[0]) :
        workQueueData.put((h5Dataset, rootpath, np.s_[ii, :], dataShape[1:],
                           data[ii*batchSize:(ii+1)*batchSize], log))

    # stream the imagery into the buffers --
    # we are threading this for efficiency
    def readImagery() :
        while True :
            h5, rootpath, sliceIndex, imSize, imFiles, log = \
                workQueueData.get()

            # load the batch locally so our writes are coherent
            tmp = np.ndarray((imSize), theano.config.floatX)
            for ii, imFile in enumerate(imFiles) :
                tmp[ii][:] = padImageData(
                    preprocessData(rootpath, imFile[0], log=log),
                    imSize[-3:])[:]
            # write the whole batch at once
            h5[sliceIndex] = tmp[:]

            workQueueData.task_done()

    # create the workers
    for ii in range(threads) :
        thread = threading.Thread(target=readImagery)
        thread.daemon = True
        thread.start()

    # join the threads and complete
    workQueueData.join()


def filterImagery (files) :
    '''Only keep relavant files to the ingest.'''
    return [x for x in files if '.hdf5' not in x and \
                                '.pkl.gz' not in x]

def readAndDivideData(path, holdoutPercentage,
                      minTest=5, saveLabels=True, log=None) :
    '''This walks the directory structure and divides the data according to the
       user specified holdout over two "train" and "test" sets.

       NOTE: For optimization this only stores the absolute path to the imagery
    '''
    from dataset.shuffle import naiveShuffle
    from dataset.reader import mostCommonExt
    from dataset.hdf5walk import hdf5walk

    walk = lambda x : os.walk(x) if os.path.isdir(x) else hdf5walk(x)

    # read the directory structure --
    # each subdirectory becomes a label and the imagery within are examples.
    # Splitting the data per label ensures each category is represented in the
    # holdout set.
    train, test, labels = [], [], []
    for root, dirs, files in walk(path) :
        # only in unsupervised learning do we inspect the top-level directory
        if saveLabels and root == path :
            continue

        # filter out support files and check if there is ingestible imagery
        files = filterImagery (files)
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

        # use the most common type of file in the dir and exclude
        # other types (avoid generated jpegs, other junk)
        suffix = mostCommonExt(files, samplesize=50)

        # prepare the data
        items = np.asarray(
            [(os.path.join(root, file), indx) for file in files
                     if file.endswith(suffix)],
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

def writePreprocessedHDF5(filepath, holdoutPercentage=.05, minTest=5,
                          batchSize=1, saveLabels=True, log=None) :
    '''Divide the directory structure over training and testing sets. This
       optionally stores the directory names at the labels and indices for
       the dataset.

       NOTE: This supports reading from a filesystem or HDF5

       filepath          : Top-level directory containing the label directories
       holdoutPercentage : Percentage of the data to holdout for testing
       minTest           : Hard minimum on holdout if percentage is low
       batchSize         : Size of a mini-batch
       saveLabels        : Specify that labels should be stored
       log               : Logger to use
    '''
    import theano
    import threading
    import multiprocessing
    from six.moves import queue
    from dataset.reader import getImageDims, mostCommon
    from dataset.hdf5 import createHDF5Labeled, createHDF5Unlabeled

    rootpath = os.path.abspath(filepath)
    outDir = rootpath if os.path.isdir(rootpath) \
             else os.path.dirname(rootpath)
    unlabeledStr = '' if saveLabels else 'un'
    outputFile = os.path.join(
        outDir, 
        os.path.splitext(os.path.basename(rootpath))[0] + 
        '_' + unlabeledStr + 'labeled_holdout_' +
        str(holdoutPercentage) + '_batch_' + str(batchSize) + '.hdf5')
    if os.path.exists(outputFile) :
        if log is not None :
            log.info('HDF5 exists for this dataset [' + outputFile +
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
    h5File = os.path.join(rootpath, os.path.split(rootpath)[1] + '.hdf5')
    if os.path.exists(h5File) :
        rootpath = h5File

    trainDir = os.path.join(rootpath, 'train')
    testDir = os.path.join(rootpath, 'test')
    if os.path.isdir(trainDir) and os.path.isdir(testDir) :

        # run each directory to grab the imagery
        trainSet = readAndDivideData(trainDir, 0., minTest=0,
                                     saveLabels=saveLabels, log=log)
        testSet = readAndDivideData(testDir, 1., minTest=0,
                                    saveLabels=saveLabels, log=log)

        # verify the labels overlap
        for label in testSet[-1] :
            if label not in trainSet[-1] :
                raise ValueError('Train and Test sets have non-overlapping ' +
                                 'labels. Please check the input directory.')

        # setup for further processing
        train, test, labels = trainSet[0], testSet[1], trainSet[-1]

    else :
        # split the data according to the user-provided holdout
        train, test, labels = readAndDivideData(
                rootpath, holdoutPercentage, minTest=minTest,
                saveLabels=saveLabels, log=log)

    if len(train) == 0 :
        raise ValueError('No training examples found [' + filepath + ']')

    # create a hdf5 memmap --
    # Here we create the handles to the data buffers. This operates on the
    # assumption the dataset may not fit entirely in memory. The handles allow
    # data to overflow and ultimately be stored entirely on disk. 
    #
    # Sample the directory for a probable chip size
    imageShape = list(mostCommon(
        [t[0] for t in train],
        lambda f : preprocessData(rootpath, f, log).shape,
        sampleSize=50))

    # Compute dataset shapes
    trainShape = [len(train) // batchSize, batchSize] + imageShape
    testShape = [len(test) // batchSize, batchSize] + imageShape

    # check for empty datasets before creating the file
    if len(train) == 0 :
        raise ValueError('No imagery in training set.')
    if len(test)  == 0 :
        raise ValueError('No imagery in testing set.')

    # check if the batch size was too large
    if len(train) > 0 and trainShape[0] == 0 :
        raise ValueError('No training batches available. Consider reducing '
                         'batchSize or adding more examples.')
    if len(test) > 0 and testShape[0] == 0 :
        raise ValueError('No testing batches available. Consider reducing '
                         'batchSize or adding more examples.')

    if log is not None :
        log.info('Writing data to HDF5')

    if saveLabels :
        [handleH5, trainDataH5, trainIndicesH5, 
         testDataH5, testIndicesH5, labelsH5] = \
            createHDF5Labeled (outputFile, 
                               trainShape, theano.config.floatX, np.int32,
                               testShape, theano.config.floatX, np.int32,
                               len(labels), log)

        # populate the indice buffers
        workQueueIndices = queue.Queue()
        workQueueIndices.put((trainIndicesH5, trainIndicesH5, train))
        workQueueIndices.put((testIndicesH5, testIndicesH5, test))

        # stream the indices into the buffers --
        # we are threading this for efficiency
        def copyIndices() :
            while True :
                dataH5, indicesH5, data = workQueueIndices.get()
                dataH5[:] = np.resize(np.asarray(data).flatten()[1::2],
                                      indicesH5.shape).astype(np.int32)[:]
                workQueueIndices.task_done()
        for ii in range(2) :
            thread = threading.Thread(target=copyIndices)
            thread.daemon = True
            thread.start()
        workQueueIndices.join()

        # stream in the label in string form
        labelsH5[:] = labels[:]

    else :
        [handleH5, trainDataH5, testDataH5] = \
            createHDF5Unlabeled (outputFile, 
                                 trainShape, theano.config.floatX, 
                                 testShape, theano.config.floatX, log=log)

    # read the image data
    threads = multiprocessing.cpu_count()
    populateDataHDF5(trainDataH5, rootpath, train, trainShape,
                     batchSize, threads, log)
    populateDataHDF5(testDataH5, rootpath, test, testShape, 
                     batchSize, threads, log)

    if log is not None :
        log.info('Flushing to disk')

    # write it to disk    
    handleH5.flush()
    handleH5.close()

    # return the output filename
    return outputFile

def reuseableIngest(filepath, holdoutPercentage=.05, minTest=5,
                    batchSize=1, saveLabels=True, log=None) :
    '''Load the labeled dataset into memory. This is formatted such that the
       directory structure becomes the labels, and all imagery within the 
       directory will be assigned this label. All images in any directory is
       required to have the same dimensions.

       filepath          : This can be a cPickle, a path to the directory
                           structure.
       shared            : Load data into shared variables for training --
                           NOTE: this is only a user suggestion. However the
                                 size of the data will ultimately determine
                                 how its loaded.
       holdoutPercentage : Percentage of the data to holdout for testing
       minTest           : Hard minimum on holdout if percentage is low
       batchSize         : Size of a mini-batch
       saveLabels        : Specify that labels should be stored
       log               : Logger for tracking the progress
       return :
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
    from dataset.hdf5 import readHDF5

    if not os.path.exists(filepath) :
        raise ValueError('The path specified does not exist.')

    def testHDF5Preprocessed(filepath) :
        # check if this is a HDF5 "preprocessed" file
        if os.path.isfile(filepath) :
            fileLower = filepath.lower()
            if fileLower.endswith('.hdf5') or fileLower.endswith('.h5') :
                import h5py
                with h5py.File(filepath, mode='r') as h5 :
                    if 'train/data' in h5 and 'test/data' in h5 :
                        return False
        # otherwise we need to build the file
        return True

    # read the directory structure and pickle it up
    if testHDF5Preprocessed(filepath) :
        filepath = writePreprocessedHDF5(filepath=filepath, 
                                         holdoutPercentage=holdoutPercentage,
                                         minTest=minTest,
                                         batchSize=batchSize,
                                         saveLabels=saveLabels,
                                         log=log)

    # Load the dataset to memory
    return readHDF5(filepath, log)
