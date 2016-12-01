import os
import h5py
import numpy as np

def createHDF5Unlabeled (outputFile, trainDataShape, trainDataDtype,
                         trainMaxShape=None, log=None) :
    '''Utility to create the HDF5 file and return the handles. This allows
       users to fill out the buffers in a memory conscious manner.

       outputFile        : Name of the file to write. The extension should be
                           either .h5 or .hdf5
       trainDataShape    : Training data dimensions 
       trainDataDtype    : Training data dtype
       trainMaxShape     : Optionally specify a maxshape for the training set
       log               : Logger to use
    '''
    if not outputFile.endswith('.h5') and not outputFile.endswith('.hdf5') :
        raise Exception('The file must end in the .h5 or .hdf5 extension.')

    # create a file in a standard way
    hdf5 = h5py.File(outputFile, libver='latest', mode='w')

    # the file will always have training data in train/data
    trainData = hdf5.create_dataset('train/data', shape=trainDataShape,
                                    dtype=trainDataDtype,
                                    maxshape=trainMaxShape)

    # TODO: Should we additionally have a test/data set to allow early
    #       stoppage? This will test the ability to reconstruct data 
    #       never before encountered. It's likely a better way to perform
    #       training instead of naive number of epochs.
    return [hdf5, trainData]


def createHDF5Labeled (outputFile,
                       trainDataShape, trainDataDtype, trainIndicesDtype,
                       testDataShape, testDataDtype, testIndicesDtype, 
                       labelsShape, log=None) :
    '''Utility to create the HDF5 file and return the handles. This allows
       users to fill out the buffers in a memory conscious manner.

       outputFile        : Name of the file to write. The extension should be
                           either .h5 or .hdf5
       trainDataShape    : Training data dimensions 
       trainDataDtype    : Training data dtype
       trainIndicesDtype : Training indicies dtype
       testDataShape     : Testing data dimensions 
       testDataDtype     : Testing data dtype
       testIndicesDtype  : Testing indicies dtype
       labelsShape       : Labels shape associated with indices
       log               : Logger to use
    '''
    hdf5, trainData = createHDF5Unlabeled(outputFile, trainDataShape,
                                          trainDataDtype, log=log)

    # supervised learning will have indices associated with the training data
    trainIndices = hdf5.create_dataset('train/indices',
                                       shape=tuple(trainDataShape[:2]),
                                       dtype=trainIndicesDtype)

    # add testing data and indices
    testData = hdf5.create_dataset('test/data', shape=testDataShape,
                                   dtype=testDataDtype)
    testIndices = hdf5.create_dataset('test/indices',
                                      shape=tuple(testDataShape[:2]),
                                      dtype=testIndicesDtype)

    # each index with have an associated string label
    labelsShape = labelsShape if isinstance(labelsShape, tuple) else\
                  (labelsShape, )
    labelsDtype = h5py.special_dtype(vlen=str)
    labels = hdf5.create_dataset('labels', shape=labelsShape, 
                                 dtype=labelsDtype)

    return [hdf5, trainData, trainIndices, testData, testIndices, labels]

def writeHDF5 (outputFile, trainData, trainIndices=None, 
               testData=None, testIndices=None, labels=None, log=None) :
    '''Utility to write a hdf5 file to disk given the data exists in numpy.

       outputFile   : Name of the file to write. The extension should be pkl.gz
       trainData    : Training data (numBatch, batchSize, chan, row, col)
       trainIndices : Training indices (either one-hot or float vectors)
       testData     : Testing data (numBatch, batchSize, chan, row, col)
       testIndices  : Testing indices (either one-hot or float vectors)
       labels       : String labels associated with indices
       log          : Logger to use
    '''
    if log is not None :
        log.debug('Writing to [' + outputFile + ']')

    # write the data to disk -- if it was supplied
    with h5py.File(outputFile, mode='w') as hdf5 :
        hdf5.create_dataset('train/data', data=trainData)

        # TODO: This should also be updated if we find unsupervised training
        #       is better with early stoppage via a test set.
        if trainIndices is not None :
            hdf5.create_dataset('train/indices', data=trainIndices)
        if testData is not None :
            hdf5.create_dataset('test/data', data=testData)
        if testIndices is not None :
            hdf5.create_dataset('test/indices', data=testIndices)
        if labels is not None :
            hdf5.create_dataset('labels', data=[l.encode("ascii", "ignore") \
                                               for l in labels])

        # ensure it gets to disk
        hdf5.flush()
        hdf5.close()

def readHDF5 (inFile, log=None) :
    '''Utility to read a pickle in from disk.

       inFile : Name of the file to read. The extension should be pkl.gz
       log    : Logger to use

       return : (train, test, labels)
    '''
    if not inFile.endswith('.h5') and not inFile.endswith('.hdf5') :
        raise Exception('The file must end in the .h5 or .hdf5 extension.')

    if log is not None :
        log.debug('Opening the file in memory-mapped mode')

    # open the file
    hdf5 = h5py.File(inFile, mode='r')

    # read the available information
    trainData = hdf5.get("train/data")
    trainIndices = None
    if 'train/indices' in hdf5 :
        trainIndices = hdf5.get('train/indices')
    testData = None
    if 'test/data' in hdf5 :
        testData = hdf5.get('test/data')
    testIndices = None
    if 'test/indices' in hdf5 :
        testIndices = hdf5.get('test/indices')
    labels = None
    if 'labels' in hdf5 :
        labels = hdf5.get('labels')

    # the returned information should be checked for None
    return [trainData, trainIndices], [testData, testIndices], labels

def archiveDirToHDF5(outputFile, inDir, flushRate=1024, log=None) :
    '''Convert a directory to an HDF5 file with matching content.
       NOTE: This stores the imagery into a raw array as returned from 
             dataset.reader.openImage

       outputFile : Name of the file to write.
                    The extension should be .hdf5 or .h5
       inDir      : The direcotry to archive. All contents and their positions
                    will be stored relative to this directory.
       flushRate  : Number of files to write before forcing the flush to disk.
       log        : Logger to use
    '''
    from dataset.reader import openImage
    log.info('Archiving [' + inDir + '] into [' + outputFile + ']')

    if os.path.exists(outputFile) :
        raise ValueError('The file already exists [' + outputFile + ']')

    outLower = outputFile.lower()
    if not (outLower.endswith('.h5') or outLower.endswith('.hdf5')) :
        raise ValueError('The output file is not a valid archive [' +
                         outputFile + ']')

    # open the file and recreate the relative directory contents
    # within the HDF5 file. All images will be stored as 3-channel
    # arrays.
    with h5py.File(outputFile, 'a') as h5 :

        count = 0
        for root, dirs, files in os.walk(inDir) :

            # reset the lists to be full relative path
            dirs = [os.path.relpath(
                    os.path.join(root, dir), inDir) for dir in dirs]

            # create h5 groups for each relative directory
            [h5.create_group(dir) for dir in dirs]

            # create h5 dataset for each relative directory
            for file in files :

                # try to read the file as an image --
                # this excludes all files not readable as an image
                try :
                    image = openImage(os.path.join(root, file), log)
                except Exception as ex:
                    log.info('File type is not supported. Excluding file [' +
                             file + ']')
                    continue

                # create h5 dataset for each image --
                # this save the imagery in a three channel format
                # the image is saved into a relative path under the same name
                h5.create_dataset(
                    os.path.relpath(os.path.join(root, file), inDir),
                    data=image, compression='gzip')

                # flush the data to disk periodically
                count += 1
                if count % flushRate == 0 : 
                    log.debug('Flushing HDF5 file to disk')
                    h5.flush()

        # one final flush
        h5.flush()


def expandHDF5(hdf5, outDir, log=None) :
    '''Expand an existing HDF5 to the filesystem. This is mostly for
       debugging and verification purposes. The files will be recreate exactly
       as they are within the HDF5 archive, however the numpy arrays are
       written as .npy files.

       hdf5      : HDF5 to expand out into the file system. Extension should be
                   .hdf5 or .h5
                   NOTE: this assume the file system is sufficiently large.
       outDir    : The direcotry to place thev archive contents.
       log       : Logger to use
    '''
    log.info('Expanding [' + hdf5 + '] into [' + outDir + ']')

    def createtree(name, item) :
        outPath = os.path.join(outDir, name)

        # create the directory path
        if isinstance(item, h5py.Group) :
            os.makedirs(outPath)

        # write the image to an npy file --
        # TODO: this incurs an additional copy due to the slice
        elif isinstance(item, h5py.Dataset) :
            np.frombuffer(item[:], dtype=item.dtype).tofile(outPath)

    with h5py.File(inputfile, 'r') as h5 :
        h5.visititems(createTree)
