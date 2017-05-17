import os
import numpy as np
import theano as t

def mostCommon (arr, func, sampleSize=None) :
    '''Identify the most common element of the series.'''
    from numpy.random import choice
    from collections import Counter

    # sample the list (choose without replacement)
    arr = list(arr)
    if sampleSize is not None and sampleSize < len(arr) :
        arr = choice(arr, sampleSize, replace=False)

    # return the most common element
    counter = Counter([func(s) for s in arr])
    return counter.most_common()[0][0]

def mostCommonExt (files, samplesize=None) :
    '''Returns the most common extension in the set of names.'''
    return mostCommon(files, lambda f: os.path.splitext(f)[1], samplesize)

def atleastND (imgData, nd) :
    '''Convert input to an n-d numpy array'''
    # NOTE: np.atleast_3d does not provide enough control,
    # so this is performed manually.
    while len(imgData.shape) < nd:
        imgData = np.expand_dims(imgData, axis=0)
    return imgData

def padImageData(imgData, dims) :
    '''Add zeropadding to an image to achieve the target dimensions.'''
    if imgData.shape[-2:] != tuple(dims[-2:]) :
         # Only pad the last 2 dimensions
        pads = ((0,) * imgData.shape[:-2]) + \
               tuple([(0, a - b) for a, b in zip(dims[-2:], imgData.shape[-2:])])
        imgData = np.pad(imgData, pads, mode='constant', constant_values=0)
    return imgData

def normalize(v) :
    '''Normalize a vector in a naive manner.'''
    minimum, maximum = np.amin(v), np.amax(v)
    return (v - minimum) / (maximum - minimum)

def statisticalNorm(v) :
    '''Zero-mean and unit variance.'''
    stddev = v.std()
    if stddev == 0.:
        return v
    else:
        return normalize((v - v.mean()) / v.std())

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
    return np.resize(a, (2, imageDims[-2], imageDims[-1]))

'''TODO: These methods may need to be implemented as derived classes.'''
def openSICD(image, log=None) :
    '''This method reads the XML and complex data from SICD.'''
    import pysix.six_sicd

    # setup the schema validation if the user has it specified
    schemaPaths = pysix.six_sicd.VectorString()
    if 'SIX_SCHEMA_PATH' in os.environ :
        schemaPaths.push_back(os.environ['SIX_SCHEMA_PATH'])

    # read the image components --
    # wbData    : the raw IQ image data
    # cmplxData : the struct for sicd metadata
    wbData, cmplxData = pysix.six_sicd.read(image, schemaPaths)
    return (wbData, cmplxData)

def openSIDD(image, log=None) :
    '''Read an SIDD NITF 2.1 file and return as np.ndarray'''
    raise NotImplementedError('Implement the dataset.reader.readSIDD() method')

def openSIO(image, log=None) :
    '''Read an SIO file and return as np.ndarray'''
    import coda.sio_lite
    return coda.sio_lite.read(image)

def openNPY(image, log=None) :
    '''Read numpy array written to disk via the numpy API.'''
    return np.load(image)

def openNITF(image, log=None) :
    '''Read an NITF file and return as np.ndarray'''
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
    a = np.resize(a, (segment.subheader.getBandCount(),
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
        a = np.asarray(np.concatenate(
                       [np.asarray(x) for x in img.split()]),
                       dtype=t.config.floatX)
        a = np.resize(a, (3, img.size[1], img.size[0]))
        return a if img.mode == 'RGB' else a[[0,2,1],:,:]
    elif img.mode == 'L' :
        # just one channel
        a = np.asarray(img.getdata(), dtype=t.config.floatX)
        return np.resize(a, (1, img.size[1], img.size[0]))

def openPILImage(image, log=None) :
    '''This method should be used for all regular image formats from JPEG,
       PNG, TIFF, etc. A PIL error may originate from this method if the image
       format is unsupported.
    '''
    from PIL import Image
    img = Image.open(image)
    img.load() # because PIL can be lazy
    return makePILImageBandContiguous(img)

def openImage(image, log=None) :
    '''Load the image into memory --
       All images returned in (numChannels, numRows, numCols) format
    '''
    if log is not None :
        log.debug('Openning Image [' + image + ']')

    # TODO: Images are named differently and this is not a great measure for
    #       for image types. Change this in the future to a try/catch and 
    #       allow the library decide what it is (or pass in a parameter for
    #       optimized performance).
    imageLower = image.lower()
    if imageLower.endswith('.sio') :
        ret = openSIO(image, log)
    elif imageLower.endswith('.npy') :
        ret = openNPY(image, log)
    elif 'sicd' in imageLower :
        ret = openSICD(image, log)[0]
    elif 'sidd' in imageLower :
        ret = openSIDD(image, log)
    elif imageLower.endswith('.nitf') or imageLower.endswith('.ntf') :
        ret = openNITF(image, log)
    else :
        ret = openPILImage(image, log)

    return atleastND(ret, 3)

def preProcImage(image, log=None) :
    '''This reads and prepares the image for training. 
       NOTE: The remap is based on the datatype in the np.ndarray.

       image: input image - str or np.ndarray accepted
       return: output image remapped for training/testing format --
               (numChannels, numRows, numCols)
    '''
    imageBuf = image if not isinstance(image, str) else openImage(image, log)
    return convertPhaseAmp(imageBuf, log) if imageBuf.dtype == np.complex64 \
           else statisticalNorm(imageBuf)

def getImageDims(image, log=None) :
    '''Load the image and return its dimensions.
        format -- (numChannels, rows, cols)
    '''
    return preProcImage(image, log).shape
