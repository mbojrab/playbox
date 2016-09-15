import h5py

def createHDF5Unlabeled (outputFile, trainDataShape, trainDataDtype, log=None):
    '''Utility to create the HDF5 file and return the handles. This allows
       users to fill out the buffers in a memory conscious manner.

       outputFile        : Name of the file to write. The extension should be
                           either .h5 or .hdf5
       trainDataShape    : Training data dimensions 
       trainDataDtype    : Training data dtype
       log               : Logger to use
    '''
    if not outputFile.endswith('.h5') and not outputFile.endswith('.hdf5') :
        raise Exception('The file must end in the .h5 or .hdf5 extension.')
    hdf5 = h5py.File(outputFile, libver='latest', mode='w')

    trainData = hdf5.create_dataset('train/data',
                                    shape=trainDataShape, dtype=trainDataDtype)
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
                                          trainDataDtype, log)

    trainIndices = hdf5.create_dataset('train/indices',
                                       shape=tuple(trainDataShape[:2]),
                                       dtype=trainIndicesDtype)

    testData = hdf5.create_dataset('test/data', shape=testDataShape,
                                   dtype=testDataDtype)
    testIndices = hdf5.create_dataset('test/indices',
                                      shape=tuple(testDataShape[:2]),
                                      dtype=testIndicesDtype)

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
        log.info('Writing to [' + outputFile + ']')

    # write the data to disk -- if it was supplied
    with h5py.File(outputFile, mode='w') as hdf5 :
        hdf5.create_dataset('train/data', data=trainData)
        if trainIndices is not None :
            hdf5.create_dataset('train/indices', data=trainIndices)
        if testData is not None :
            hdf5.create_dataset('test/data', data=testData)
        if testIndices is not None :
            hdf5.create_dataset('test/indices', data=testIndices)
        if labels is not None :
            hdf5.create_dataset('labels', data=labels)

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
        log.info('Opening the file in memory mapped mode')

    hdf5 = h5py.File(inFile, mode='r')

    trainData = hdf5.get("train/data")
    if 'train/indices' in hdf5 :
        trainIndices = hdf5.get('train/indices')
    if 'test/data' in hdf5 :
        testData = hdf5.get('test/data')
    if 'test/indices' in hdf5 :
        testIndices = hdf5.get('test/indices')
    if 'labels' in hdf5 :
        labels = hdf5.get('labels')

    return (trainData, trainIndices), (testData, testIndices), labels
