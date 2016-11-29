import os
import numpy as np

def hdf5Dataset(filepaths, batchSize=1, log=None, chipFunc=None, **kwargs) :
    '''Create a pickle out of a directory structure. The directory structure
       is assumed to be a series of directories, each contains imagery assigned
       the label of the directory name.

       filepaths         : path to the top-level directory contain the images.
                           This can be a list of directory to find multiple
                           files.
       batchSize         : Size of a mini-batch
       log               : Logger to use
       chipFunc          : Chipping function to use
       **kwargs          : Chipping function parameters
    '''
    from dataset.hdf5 import createHDF5Unlabeled
    from dataset.shuffle import naiveShuffle

    # place the hdf5 archive in the root directory
    rootpath = os.path.commonprefix(filepaths)

    # get the image dimensions
    images = os.listdir(filepaths[0])
    if len(images) > 0 :
        from dataset.reader import getImageDims
        imageShape = getImageDims(os.path.join(filepaths[0], images[0]))
    else :
        raise ValueError('No files found in [' + filepaths[0] + ']')

    # setup the filename according to the parameters
    if chipFunc is not None :
        chipName = '_chip_' + chipFunc.__name__
        chipSize = kwargs['chipSize']
    else :
        chipName = ''
        chipSize = imageShape[-2:]
    chipSizeStr = '_' + str(chipSize[0]) + 'x' + str(chipSize[1])
    outputFile = os.path.join(rootpath, os.path.basename(rootpath) +
                                  '_unlabeled' + chipName + chipSizeStr +
                                  '_batch_' + str(batchSize) + '.hdf5')

    if os.path.exists(outputFile) :
        if log is not None :
            log.info('HDF5 exists for this dataset [' + outputFile +
                     ']. Using this instead.')
        return outputFile

    # open the HDF5 file --
    # HDF5 has the ability to be dynamically scaled. In this particular case
    # we don't know the number of files yet, especially if chipFunc was
    # specified, so this allows the file to scale to the appropriate size.
    [handleH5, trainDataH5] = createHDF5Unlabeled(
        outputFile, imageShape, np.float32, 
        tuple([None, None] + list(chipSize)), log)

    # read the directory
    images = []
    for filepath in filepaths :
        images.extend([os.path.join(filepath, im) \
                       for im in os.listdir(filepath)])

    # randomize the data across categories -- otherwise its not stochastic --
    # NOTE: this is only pseudo-random because we are only randomizing the
    #       files. If the user specifies a chipping utility these images will
    #       be grouped contiguously. If the user desires complete stochasticity
    #       they can chip the imagery to disk, and then use this function.
    if log is not None :
        log.info('Shuffling the data for randomization')
    naiveShuffle(images)

    # walk through each file and either batch it directly or chip it according
    # to the user specified chipping utility.
    if chipFunc is not None :
        # TODO: insert logic for chipping the imagery individually and 
        #       batching appropriattely
        raise NotImplementedError('Implement support for chipped ingest.')
        #from dataset.reader import preProcImage
        #from dataset.chip import prepareChips
        #
        #for im in images :
        #    try :
        #        # chip the image using the provided utility
        #        chips = np.asarray(chipFunc(preProcImage(im, log), **kwargs))
        #    except IOError : pass
    else :
        from dataset.ingest.labeled import readDataset
        from multiprocessing import cpu_count

        # read all imagery directly --
        # this assume all imagery is of the same size
        readDataset(trainDataH5, images, imageShape, batchSize,
                    cpu_count(), log)

    if log is not None :
        log.info('Flushing to disk')

    # write it to disk    
    handleH5.flush()
    handleH5.close()

    # return the output filename
    return outputFile


def ingestImagery(filepaths, shared=True, batchSize=1,
                  log=None, chipFunc=None, **kwargs) :
    '''Load the unlabeled dataset into memory. This reads and chips any
       imagery found within the filepath according the the options sent to the
       function.

       filepath : This can be a hdf5, a path to the directory structure, of a 
                  list of directories containing files.
       shared   : Load data into shared variables for training
       batchSize: Size of a mini-batch
       log      : Logger for tracking the progress
       chipFunc : Chipping utility to use on each image
       kwargs   : Parameters specific for the chipping function
       return   : (trainingData, pixelRegion={None})
    '''
    import theano.tensor as t
    from dataset.pickle import readPickleZip
    from dataset.ingest.labeled import checkAvailableMemory

    if not isinstance(filepaths, list) :
        filepaths = [filepaths]
    filepaths = [os.path.abspath(d) for d in filepaths]

    # verify all paths exist
    for filepath in filepaths :
        if not os.path.exists(filepath) :
            raise ValueError('The path specified does not exist.')

    # read the directory structure and chip it
    if os.path.isdir(filepaths[0]) :
        filepath = hdf5Dataset(filepaths, batchSize=batchSize, log=log,
                               chipFunc=chipFunc, **kwargs['kwargs'])
    else :
        filepath = filepaths[0]

    # Load the dataset to memory
    train = readPickleZip(filepath, log)

    # calculate the memory needed by this dataset
    dt = 4. if t.config.floatX == 'float32' else 8.
    dataMemoryConsumption = np.prod(np.asarray(
        train.shape, dtype=np.float32)) * dt

    # check physical memory constraints
    shared = checkAvailableMemory(dataMemoryConsumption, shared, log)

    # load each into shared variables -- 
    # this avoids having to copy the data to the GPU between each call
    if shared is True :
        from dataset.shared import toShared
        if log is not None :
            log.debug('Transfer the memory into shared variables')
        try :
            return toShared(train)
        except :
            pass
    return train
