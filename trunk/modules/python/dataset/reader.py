import os
import numpy as np
import theano as t

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

def readSICD(image, log=None) :
    '''This method should read and prepare the data for training or testing.'''
    wbData, cmplxData = openSICD(image, log)
    return convertPhaseAmp(wbData, log)

def readSIDD(image, log=None) :
    '''This method should read a prepare the data for training or testing.'''
    raise NotImplementedError('Implement the datasetUtils.readSIDD() method')

def openSIO(image, log=None) :
    import coda.sio_lite
    return coda.sio_lite.read(image)

def readSIO(image, log=None) :
    imData = openSIO(image, log)
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
