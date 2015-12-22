import theano, numpy, cPickle, gzip, os
import theano.tensor as t

def loadShared(x, borrow=True) :
    if not isinstance(x, numpy.ndarray) :
        x = numpy.asarray(x, dtype=theano.config.floatX)
    return theano.shared(x, borrow=borrow)
def splitToShared(x, borrow=True) :
    data, label = x
    return (loadShared(data), t.cast(loadShared(label), 'int32'))
def ingestImagery(filepath=None, log=None) :
    '''Load the dataset provided by the user.
       filepath : This can be a cPickle, a path to the directory structure,
                  or None if the MNIST dataset should be loaded.
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
    if log is not None :
        log.debug('Transfer the memory into shared variables')
    return splitToShared(train), splitToShared(test), labels


def readImage(image, log=None) :
    '''Load the image into memory. It can be any type supported by PIL
    '''
    from PIL import Image
    if log is not None :
        log.debug('Openning Image [' + image + ']')
    img = Image.open(image)
    if img.mode == 'RBG' :
        # channels are interleaved by band
        a = numpy.concatenate((img.split()), dtype=theano.config.floatX)
        return numpy.resize(a, (3, img.size[1], img.size[0]))
    elif img.mode == 'L' :
        # just one channel
        a = numpy.asarray(img.getdata(), dtype=theano.config.floatX)
        return numpy.resize(a, (1, img.size[1], img.size[0]))
def makeMiniBatch(x, log=None) :
    '''Deinterleave the data and labels. Resize so we can use batched learning
    '''
    numImages = len(x)
    if numImages == 0 :
        raise Exception('No images were found.')

    # make a mini-batch of size (numImages, numChannels, rows, cols) --
    # NOTE: We assume all imagery is of the same dimensions  
    numChan, rows, cols = x[0][0].shape[0], x[0][0].shape[1], x[0][0].shape[2]
    temp = numpy.concatenate(x)
    tempData = numpy.resize(numpy.concatenate(temp[::2]),
                            (numImages, numChan, rows, cols))

    # labels are now just contiguous
    tempLabel = numpy.asarray(temp[1::2], dtype='int32')
    return tempData, tempLabel
def pickleDataset(filepath, holdoutPercentage=.05, minTest=5, log=None) :
    '''Create a pickle out of a directory structure. The directory structure
       is assumed to be a series of directories, each contain imagery assigned
       assigned the label of the directory name.

       filepath : path to the top-level directory contain the label directories
    '''
    import random

    rootpath = os.path.abspath(filepath)
    outputFile = os.path.join(rootpath, os.path.basename(rootpath) + 
                              '_holdout_' + str(holdoutPercentage) +
                              '.pkl.gz')
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
        # 
        numTest = max(minTest, int(holdoutPercentage * len(files)))
        if log is not None :
            log.debug('Holding out [' + str(numTest) + '] of [' + \
                      str(len(files)) + ']')
        for ii in range(len(files)) :
            imgLabel = readImage(os.path.join(root, files[ii]), log), labelIndx
            if ii % numTest == 0 : 
                train.append(imgLabel)
            else :
                test.append(imgLabel)

    # randomize the data -- otherwise its not stochastic
    if log is not None :
        log.info('Shuffling the data for randomization')
    random.shuffle(train)
    random.shuffle(test)

    # make it a contiguous buffer, so we have the option of batch learning
    if log is not None :
        log.info('Create the mini-batches')
    train = makeMiniBatch(train)
    test = makeMiniBatch(test)

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
