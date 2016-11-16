import os
import numpy as np
import theano as t

def mostCommon(arr, func, sampleSize=None) :
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

def mostCommonExt(files, samplesize=None) :
    '''Returns the most common extension in the set of names.'''
    return mostCommon(files, lambda f: os.path.splitext(f)[1], samplesize)

def atleastND(imgData, nd) :
    '''Convert input to an n-d numpy array'''
    # NOTE: np.atleast_3d does not provide enough control,
    # so this is performed manually.
    while len(imgData.shape) < nd:
        imgData = np.expand_dims(imgData, axis=0)
    return imgData

def padImageData(imgData, dims) :
    '''Add zeropadding to an image to achieve the target dimensions.'''
    if imgData.shape != tuple(dims) :
        pads = tuple([(0, a - b) for a, b in zip(dims, imgData.shape)])
        imgData = np.pad(imgData, pads, mode='constant', constant_values=0)
    return imgData

def normalize(v) :
    '''Normalize a vector in a naive manner.'''
    minimum, maximum = np.amin(v), np.amax(v)
    return (v - minimum) / (maximum - minimum)

def statisticalNorm(v) :
    '''Zero-mean and unit variance.'''
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
    return np.resize(a, (2, imageDims[0], imageDims[1]))

'''TODO: These methods may need to be implemented as derived classes.'''
def openSICD(image, log=None) :
    '''This method reads the XML and complex data from SICD.'''
    import pysix.six_sicd

    if type(image) != numpy.ndarray:
        return image

    # setup the schema validation if the user has it specified
    schemaPaths = pysix.six_sicd.VectorString()
    if 'SIX_SCHEMA_PATH' in os.environ :
        schemaPaths.push_back(os.environ['SIX_SCHEMA_PATH'])

    # read the image components --
    # wbData    : the raw IQ image data
    # cmplxData : the struct for sicd metadata
    wbData, cmplxData = pysix.six_sicd.read(image, schemaPaths)
    return (wbData, cmplxData)

def readSICD(image, log=None, **kwargs) :
    '''This method should read and prepare the data for training or testing.'''
    wbData, cmplxData = openSICD(image, log)
    raw = kwargs.get('raw', False)
    if not raw:
        wbData = convertPhaseAmp(wbData, log)
    else:
        wbData = wbData, 'sicd'
    return wbData

def readSIDD(image, log=None, **kwargs) :
    '''This method should read a prepare the data for training or testing.'''
    raise NotImplementedError('Implement the datasetUtils.readSIDD() method')

def openSIO(image, log=None) :
    import coda.sio_lite
    if type(image) == np.ndarray:
        return image
    return coda.sio_lite.read(image)

def transformSIOData(imData, log=None):
    ''' Apply phase amp conversion to image data '''
    if imData.dtype == np.complex64 :
        return convertPhaseAmp(imData, log)
    else :
        # TODO: this assumes the imData is already band-interleaved
        return imData

def readSIO(image, log=None, **kwargs) :
    siodata = openSIO(image, log)
    raw = kwargs.get('raw', False)
    if not raw:
        siodata = transformSIOData(siodata, log)
    else:
        siodata = siodata, '.sio'
    return siodata

def readerToNumpyArray(reader):
    # this is the bottom half of the sio_lite.read function
    # TODO: patch coda so that this is avalable everywhere
    from coda.sio_lite import dtypeFromSioType
    header = reader.getHeader()

    elementSize = header.getElementSize()
    dtype = dtypeFromSioType(header.getElementType(), elementSize)

    numpyArray = np.empty(shape=(header.getNumLines(),
                                 header.getNumElements()),
                          dtype=dtype)
    pointer, ro = numpyArray.__array_interface__['data']
    reader.read(pointer,
                numpyArray.shape[0] * numpyArray.shape[1] * elementSize)
    return numpyArray

def memSIOToNumpy(buf):
    ''' Read an sio byte buffer into a numpy array '''
    from coda.sio_lite import StreamReader
    from coda.coda_io import StringStream

    inputstream = StringStream()
    inputstream.write(buf)
    reader = StreamReader(inputstream)
    return readerToNumpyArray(reader)

def readSIOFromMem(image, log=None) :
    return transformSIOData(memSIOToNumpy(image), log)

def readNITF(image, log=None, **kwargs) :
    import nitf

    if type(image) != np.ndarray: 
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

        a = np.resize(a, 
                      (segment.subheader.getBandCount(),
                       window.numRows, window.numCols))

        # explicitly close the handle -- for peace of mind
        reader.io.close()
    else:
        a = image[0]

    raw = kwargs.get('raw', False)
    if not raw:
        a = statisticalNorm(a)
    else:
        a = a, '.nitf'
    return a

def makePILImageBandContiguous(img, log=None) :
    '''This will split the image so each channel will be contigous in memory.
       The resultant format is (channels, rows, cols), where channels are
       ordered in [Red, Green. Blue] for three channel products.
    '''
    if img.mode == 'RBG' or img.mode == 'RGB' :
        # channels are interleaved by band
        a = np.asarray(np.concatenate(
                       [statisticalNorm(np.asarray(x)) for x in img.split()]),
                       dtype=t.config.floatX)
        a = np.resize(a, (3, img.size[1], img.size[0]))
        return a if img.mode == 'RGB' else a[[0,2,1],:,:]
    elif img.mode == 'L' :
        # just one channel
        a = np.asarray(img.getdata(), dtype=t.config.floatX)
        return np.resize(statisticalNorm(a), (1, img.size[1], img.size[0]))

def readPILImage(image, log=None, **kwargs) :
    '''This method should be used for all regular image formats from JPEG,
       PNG, TIFF, etc. A PIL error may originate from this method if the image
       format is unsupported.
    '''
    from PIL import Image
    from io import BytesIO

    # PIL images can be a bunch of different types
    # with different memory layouts that are nasty
    # to recreate (putting them as numpy arrays
    # disturbs their delicate nature)
    #
    # So for 'raw' input, just punt on interpreting
    # them at all and just grab the raw file bytes.
    # 
    # When we need the 'processed' version, PIL
    # is smart enough to decode these in memory

    if type(image) == tuple:
        # pass in buffer and type as tuple
        # read raw bytes
        img = Image.open(BytesIO(image[0].tobytes()))
    else:
        img = Image.open(image)
        img.load() # because PIL can be lazy

    raw = kwargs.get('raw', False)
    if not raw:
        out = makePILImageBandContiguous(img, log)
    else:
        if type(image) == str:
            out = (np.fromfile(image, dtype=np.uint8),
                   img.mode)
        else:
            # just hand back the input
            out = image

    return out

def readImage(image, log=None, **kwargs) :
    '''Load the image into memory. It can be any type supported by PIL.'''
    if log is not None :
        log.debug('Opening Image [' + image + ']')
    if type(image) == tuple:
        typeinfo = image[1]  # type of image is second tuple entry
        image = image[0]
        pilargs = (image, typeinfo)
    else:
        typeinfo = image.lower()
        pilargs = image

    # TODO: Images are named differently and this is not a great measure for
    #       for image types. Change this in the future to a try/catch and 
    #       allow the library decide what it is (or pass in a parameter for
    #       optimized performance).
    ret = None
    if typeinfo.endswith('.sio') :
        ret = readSIO(image, log, **kwargs)
    elif 'sicd' in typeinfo :
        ret = readSICD(image, log, **kwargs)
    elif 'sidd' in typeinfo :
        ret = readSIDD(image, log, **kwargs)
    elif typeinfo.endswith('.nitf') or typeinfo.endswith('.ntf') :
        ret = readNITF(image, log, **kwargs)
    else :
        ret = readPILImage(pilargs, log, **kwargs)

    raw = kwargs.get('raw', False)
    return ret if raw else atleastND(ret, 3)

def getImageDims(image, log=None) :
    '''Load the image and return its dimensions.
        format -- (numChannels, rows, cols)
    '''
    return readImage(image, log).shape
