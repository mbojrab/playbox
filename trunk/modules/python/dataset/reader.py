import os
import numpy as np
import theano as t

def ingestImagery(filepath, shared=False, log=None) :
    '''Load the dataset provided by the user.
       filepath : This can be a cPickle, a path to the directory structure.
       shared   : Load data into shared variables for training
       log      : Logger for tracking the progress
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
    from dataset.shared import splitToShared
    from dataset.pickle import readPickleZip
    train = test = None

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

def normalize(v) :
    '''Normalize a vector in a naive manner.'''
    minimum, maximum = np.amin(v), np.amax(v)
    return (v - minimum) / (maximum - minimum)

def convertPhaseAmp(imData, log=None) :
    '''Extract the phase and amplitude components of a complex number.
       This is assumed to be a better classifier than the raw IQ image.
       TODO: For SAR products, the data is Rayleigh distributed for the 
             amplitude component, so it may be best to perform a more rigorous
             remap here. Amplitude recognition can be affected greatly by the
             strength of Rayleigh distribution.
       TODO: Research other possible feature spaces which could elicit better
             learning or augment the phase/amp components.
    '''
    if imData.dtype != np.complex64 :
        raise ValueError('The array must be of type numpy.complex64.')
    imageDims = imData.shape
    a = np.asarray(np.concatenate((normalize(np.angle(imData)), 
                                   normalize(np.absolute(imData)))),
                      dtype=t.config.floatX)
    return np.resize(a, (2, imageDims[0], imageDims[1]))

'''TODO: These methods may need to be implemented as derived classes.'''
def readSICD(image, log=None) :
    '''This method should read a prepare the data for training or testing.'''
    import pysix.six_sicd

    # setup the schema validation if the user has it specified
    schemaPaths = pysix.six_sicd.VectorString()
    if os.environ.has_key('SIX_SCHEMA_PATH') :
        schemaPaths.push_back(os.environ['SIX_SCHEMA_PATH'])

    # read the image components --
    # wbData    : the raw IQ image data
    # cmplxData : the struct for sicd metadata
    wbData, cmplxData = pysix.six_sicd.read(image, schemaPaths)
    return convertPhaseAmp(wbData, log)

def readSIDD(image, log=None) :
    '''This method should read a prepare the data for training or testing.'''
    raise NotImplementedError('Implement the datasetUtils.readSIDD() method')

def readSIO(image, log=None) :
    import coda.sio_lite
    imData = coda.sio_lite.read(image)
    if imData.dtype == np.complex64 :
        return convertPhaseAmp(imData, log)
    else :
        # TODO: this assumes the imData is already band-interleaved
        return imData

def readNITF(image, log=None) :
    import nitf

    # read the nitf
    reader, record = nitf.read(image)

    # there could be multiple images per nitf --
    # for now just read the first.
    # TODO: we could read each image separately, but its unlikely to
    #       encounter a multi-image file in the wild.
    segment = record.getImages()[0]
    imageReader = reader.newImageReader(0)
    window = nitf.SubWindow()
    window.numRows = segment.subheader['numRows'].intValue()
    window.numCols = segment.subheader['numCols'].intValue()
    window.bandList = range(segment.subheader.getBandCount())

    # read the bands and interleave them by band --
    # this assumes the image is non-complex and treats bands as color.
    a = np.concatenate(imageReader.read(window))

    a = np.resize(normalize(a), (segment.subheader.getBandCount(),
                                 window.numRows, window.numCols))
    # explicitly close the handle -- for peace of mind
    reader.io.close()
    return a

def makePILImageBandContiguous(img, log=None) :
    '''This will split the image so each channel will be contigous in memory.
       The resultant format is (channels, rows, cols), where channels are
       ordered in [Red, Green. Blue] for three channel products.
    '''
    if img.mode == 'RBG' or img.mode == 'RGB' :
        # channels are interleaved by band
        a = np.asarray(np.concatenate(img.split()), 
                       dtype=t.config.floatX)
        a = np.resize(normalize(a), (3, img.size[1], img.size[0]))
        return a if img.mode == 'RGB' else a[[0,2,1],:,:]
    elif img.mode == 'L' :
        # just one channel
        a = np.asarray(img.getdata(), dtype=t.config.floatX)
        return np.resize(normalize(a), (1, img.size[1], img.size[0]))

def readPILImage(image, log=None) :
    '''This method should be used for all regular image formats from JPEG,
       PNG, TIFF, etc. A PIL error may originate from this method if the image
       format is unsupported.
    '''
    from PIL import Image
    img = Image.open(image)
    img.load() # because PIL can be lazy
    return makePILImageBandContiguous(img)

def readImage(image, log=None) :
    '''Load the image into memory. It can be any type supported by PIL.'''
    if log is not None :
        log.debug('Openning Image [' + image + ']')
    imageLower = image.lower()
    # TODO: Images are named differently and this is not a great measure for
    #       for image types. Change this in the future to a try/catch and 
    #       allow the library decide what it is (or pass in a parameter for
    #       optimized performance).
    if 'sicd' in imageLower :
        return readSICD(image, log)
    elif 'sidd' in imageLower :
        return readSIDD(image, log)
    elif imageLower.endswith('.nitf') or imageLower.endswith('.ntf') :
        return readNITF(image, log)
    elif imageLower.endswith('.sio') :
        return readSIO(image, log)
    else :
        return readPILImage(image, log)

def pickleDataset(filepath, holdoutPercentage=.05, minTest=5,
                  batchSize=1, log=None) :
    '''Create a pickle out of a directory structure. The directory structure
       is assumed to be a series of directories, each contain imagery assigned
       assigned the label of the directory name.

       filepath : path to the top-level directory contain the label directories
    '''
    import os
    from minibatch import makeContiguous
    from shuffle import naiveShuffle
    from pickle import writePickleZip

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
