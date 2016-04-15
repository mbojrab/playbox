import theano, numpy, cPickle, gzip, os
import theano.tensor as t

def loadShared(x, borrow=True) :
    '''Transfer numpy.array to theano.shared variable.
       NOTE: Shared variables allow for optimized GPU execution
    '''
    if not isinstance(x, numpy.ndarray) :
        x = numpy.asarray(x, dtype=theano.config.floatX)
    return theano.shared(x, borrow=borrow)

def splitToShared(x, borrow=True) :
    '''Create shared variables for both the input and expectedOutcome vectors.
       x      : This can be a vector list of inputs and expectedOutputs. It is
                assumed they are of the same length.
                    Format - (data, label)
       borrow : Should the theano vector accept responsibility for the memory
       return : Shared Variable equivalents for these items
                    Format - (data, label)
    '''
    data, label = x
    return (loadShared(data), t.cast(loadShared(label), 'int32'))

def ingestImagery(filepath=None, shared=False, log=None) :
    '''Load the dataset provided by the user.
       filepath : This can be a cPickle, a path to the directory structure,
                  or None if the MNIST dataset should be loaded.
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

    # if None load the MNIST
    filepath = 'mnist.pkl.gz' if filepath is None else filepath
    train = test = None

    if log is not None :
        log.info('Ingesting imagery...')

    # the mnist dataset is a special case
    if 'mnist.pkl.gz' in filepath :

        # see if we have previously downloaded the file
        if filepath not in os.listdir(os.getcwd()) :
            import urllib
            url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
            if log is not None :
                log.debug('Downloading data from ' + url)
            urllib.urlretrieve(url, filepath)

        # Load the dataset to memory -- 
        # the mnist dataset is a special case created by University of Toronto
        if log is not None :
            log.debug('Load the data into memory')
        with gzip.open(filepath, 'rb') as f :
            # this dataset has a valid and test and no labels. 
            train, valid, test = cPickle.load(f)

            # add the validation set to the training set
            if log is not None :
                log.debug('Combine the Train and Valid datasets')
            trainData, trainLabel = train
            validData, validLabel = valid
            train = numpy.concatenate((trainData, validData)), \
                    numpy.concatenate((trainLabel, validLabel))

            # create a label vector
            if log is not None :
                log.debug('Create the labels')
            labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    else :
        # Load the dataset to memory
        if log is not None :
            log.debug('Load the data into memory')
        with gzip.open(filepath, 'rb') as f :
            train, test, labels = cPickle.load(f)

    # load each into shared variables -- 
    # this avoids having to copy the data to the GPU between each call
    if shared is True :
        if log is not None :
            log.debug('Transfer the memory into shared variables')
        return splitToShared(train), splitToShared(test), labels
    else :
        return train, test, labels

def normalize(v) :
    '''Normalize a vector in a naive manner.
       TODO: For SAR products, the data is Rayleigh distributed for the 
             amplitude component, so it may be best to perform a more rigorous
             remap here. Amplitude recognition can be affected greatly by the
             strength of Rayleigh distribution.
    '''
    minimum, maximum = numpy.amin(v), numpy.amax(v)
    return (v - minimum) / (maximum - minimum)

def convertPhaseAmp(imData, log=None) :
    '''Extract the phase and amplitude components of a complex number.
       This is assumed to be a better classifier than the raw IQ image.
       TODO: the amplitude components are Rayleigh distributed and may need to
             be remapped (using histoEq) to better fit the bit range. The 
             NN weights may also correct for this.
       TODO: Research other possible feature spaces which could elicit better
             learning or augment the phase/amp components.
    '''
    if imData.dtype != numpy.complex64 :
        raise ValueError('The array must be of type numpy.complex64.')
    imageDims = imData.shape
    a = numpy.asarray(numpy.concatenate((normalize(numpy.angle(imData)), 
                                         normalize(numpy.absolute(imData)))),
                      dtype=theano.config.floatX)
    return numpy.resize(a, (2, imageDims[0], imageDims[1]))

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
    if imData.dtype == numpy.complex64 :
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
    a = numpy.concatenate(imageReader.read(window))

    a = numpy.resize(normalize(a), (segment.subheader.getBandCount(),
                                    window.numRows, window.numCols))
    # explicitly close the handle -- for peace of mind
    reader.io.close()
    return a

def readPILImage(image, log=None) :
    '''This method should be used for all regular image formats from JPEG,
       PNG, TIFF, etc. A PIL error may originate from this method if the image
       format is unsupported.
    '''
    from PIL import Image
    img = Image.open(image)
    img.load() # because PIL can be lazy
    if img.mode == 'RBG' or img.mode == 'RGB' :
        # channels are interleaved by band
        a = numpy.asarray(numpy.concatenate(img.split()), 
                          dtype=theano.config.floatX)
        a = numpy.resize(normalize(a), (3, img.size[1], img.size[0]))
        return a if img.mode == 'RGB' else a[[0,2,1],:,:]
    elif img.mode == 'L' :
        # just one channel
        a = numpy.asarray(img.getdata(), dtype=theano.config.floatX)
        return numpy.resize(normalize(a), (1, img.size[1], img.size[0]))

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

def makeMiniBatch(x, batchSize=1, log=None) :
    '''Deinterleave the data and labels. Resize so we can use batched learning.
       x         : numpy.ndarray containing tuples of elements
                   [(imageData1, labelInt1), (imageData2, labelInt2),...] 
       batchSize : the size of the training mini-batch. this is intended to be
                   be an integer in range [1:numImages].
       return    : the deinterleaved data of the specified batching size.
                   [[imageData1, imageData2], [labelInt1, labelInt2]]
    '''
    import math
    numImages = len(x)
    if numImages == 0 :
        raise Exception('No images were found.')
    numBatches = int(math.floor(float(numImages) / float(batchSize)))

    # make a mini-batch of size --
    # NOTE: We assume all imagery is of the same dimensions
    numChan, rows, cols = x[0][0].shape[0], x[0][0].shape[1], x[0][0].shape[2]
    temp = numpy.concatenate(x)
    if log is not None :
        log.info('Creating Dataset : ' + 
                 str((numBatches, batchSize, numChan, rows, cols)))
    tempData = numpy.resize(numpy.concatenate(temp[::2]),
                            (numBatches, batchSize, numChan, rows, cols))

    # labels are now just contiguous
    # TODO: this needs to account for different batch sizes
    tempLabel = numpy.resize(numpy.asarray(temp[1::2], dtype='int32'),
                             (numBatches, batchSize))
    return tempData, tempLabel

def pickleDataset(filepath, holdoutPercentage=.05, minTest=5,
                  batchSize=1, log=None) :
    '''Create a pickle out of a directory structure. The directory structure
       is assumed to be a series of directories, each contain imagery assigned
       assigned the label of the directory name.

       filepath : path to the top-level directory contain the label directories
    '''
    import random

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
        # don't add it if there are no files
        if len(files) == 0 :
            if log is not None :
                log.debug('No files found in [' + root + ']')
            continue

        # add the new label for the current directory
        label = os.path.relpath(root, rootpath).replace(os.path.sep, '.')
        labels.append(label)
        labelIndx = len(labels) - 1
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
                imgLabel = readImage(os.path.join(root, files[ii]), log), labelIndx
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
    random.shuffle(train)
    random.shuffle(test)

    # make it a contiguous buffer, so we have the option of batch learning --
    #
    # Here the training set can be set to a specified batchSize. This will
    # help to speed processing and reduce high-frequency noise during training.
    #
    # Alternatively the test set does not require mini-batches, so we instead
    # make the batchSize=numImages, so we can quickly test the accuracy of
    # the network against our entire test set in one call.
    if log is not None :
        log.info('Creating the mini-batches')
    train = makeMiniBatch(train, batchSize)
    test =  makeMiniBatch(test, batchSize)

    # pickle the dataset
    if log is not None :
        log.info('Compressing to [' + outputFile + ']')
    with gzip.open(outputFile, 'wb') as f :
        f.write(cPickle.dumps((train, test, labels)))

    # return the output filename
    return outputFile

if __name__ == '__main__' :
    import logging
    log = logging.getLogger('datasetUtils')
    log.setLevel('INFO')
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    stream = logging.StreamHandler()
    stream.setLevel('INFO')
    stream.setFormatter(formatter)
    log.addHandler(stream)

    i = ingestImagery(pickleDataset('G:/coding/input/binary_smaller', log=log),
                      log=log)
