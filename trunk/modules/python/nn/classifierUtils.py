import numpy as np
from net import ClassifierNetwork

def createClassMap(network, image) :
    '''This is an exhaustive search algorithm which checks all available 
       sub-regions. This creates two matrices of the classifications and 
       their associated likelihood confidences.
       This can be thought of as a heat map for the classification and the 
       confidence in those classifications. Areas of high-confidence and large
       regions of like classifications are likely to contain an identifiable
       object.

       network : Pre-trained ClassifierNetwork to classify the image
       image   : image to classify. The size is assumed to be greater than
                 or equal to the network's input size. 
                 (numChannels, numRows, numCols)

       return  : numpy.ndarray(classification), numpy.ndarray(confidence)
    '''

    # verify the types and sizing
    if not isinstance(network, ClassifierNetwork) :
        raise ValueError('network must be a ClassifierNetwork object')
    if not isinstance(image, np.ndarray) :
        raise ValueError('imageRegion must be a numpy.ndarray object')

    # only check the (numChannels, numRows, numCols) sizing on the network
    networkInputShape = network.getNetworkInputSize()[-3:]
    if image.shape[0] != networkInputShape[0] :
        raise Exception('The imageRegion has a different number of channels ' +
                        'than the network was trained to recognize.')
    if image.shape[1] < networkInputShape[1] or \
       image.shape[2] < networkInputShape[2] :
        raise Exception('The imageRegion is smaller than the network input.')

    # create the memory buffers
    featureShape = (image.shape[1] - networkInputShape[1] + 1,
                    image.shape[2] - networkInputShape[2] + 1)
    classifications = np.ndarray(featureShape, dtype='int32')
    confidence = np.ndarray(featureShape)

    # fill out each matrix with the network output
    # TODO: It's likely faster to slide down the rows, because the reload
    #       will be aligned with contiguous memory.
    # TODO: Thread this, it's embarrassingly parallel, but watch out
    #       because the internal values may cause race conditions.
    numRows, numCols = networkInputShape[1], networkInputShape[2]
    for ii in range(featureShape[0]) :
        for jj in range(featureShape[1]) :
            classifications[ii][jj], softmax = network.classifyAndSoftmax(
                np.reshape(image[:,ii:ii+numRows,jj:jj+numCols],
                           (1,1,numRows,numCols)))
            confidence[ii][jj] = softmax[0][classifications[ii][jj]]


    # return the results
    return classifications, confidence

def singleClassify(network, image) :
    '''This is an exhaustive search algorithm which checks all available 
       sub-regions. This assumes the image or subregion contains only one 
       object, and attempts to return the most likely candidate classification
       based on the mode classification and its likelihood.  

       NOTE: The image is assumed to contain only one classification, so
             inputs containing more than one will return undefined behavior.

       network : Pre-trained ClassifierNetwork to classify the image
       image   : image to classify. The size is assumed to be greater than or
                 equal to the network's input size. 
                 (numChannels, numRows, numCols)

       return  : (classification index, likelihood value)
    '''
    from scipy.stats import mode

    # create the classification and confidence matrices
    classifications, confidence = createClassMap(network, image)

    # the mode will find the most frequently classified value --
    # we use this as the most likely candidate classification
    mostFreqClass = mode(classifications)

    # find the average confidence of the correct classifications
    # TODO: we should likely account for incorrect classificaitons in the
    #       likelihood value. More research should be performed.
    mask = np.ma.masked_equal(classifications, mostFreqClass)
    return mostFreqClass, np.ma.masked_array(confidence, mask=mask).mean()


if __name__ == "__main__" :
    import argparse
    from datasetUtils import readImage
    from net import ClassifierNetwork as Network

    parser = argparse.ArgumentParser()
    parser.add_argument('--multiple', dest='multi', type=bool, default=True,
                        help='Specify that the image contains multiple objects')
    parser.add_argument('--syn', dest='synapse', type=str, default=None,
                        help='Load from a previously saved network.')
    parser.add_argument('--out', dest='outFile', type=str, default=None,
                        help='Output image with box classifications.')
    parser.add_argument('image', help='Image to classify')
    options = parser.parse_args()

    # load everything into memory
    image = readImage(options.image)

    network = Network(options.synapse)

    # perform classification for multiple objects
    if options.multi :
        classification, confidence = createClassMap(network, image)
    else :
        classification, confidence = singleClassify(network, image)

    # write a product if it was asked for
    if options.outFile is not None : 
        from PIL import Image
        confidence = np.ndarray.astype(normalize(confidence)*255, 
                                       dtype='uint8')
        img = Image.fromarray(confidence, mode='L')
        img.save(options.outFile)
