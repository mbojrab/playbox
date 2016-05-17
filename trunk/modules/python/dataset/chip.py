
def prepareChips (chips, pixelRegion, log=None) :
    '''Make data contiguous in memory
       All pixels will be arranged into a single tensor of format
           (numChips,numChannels,chipRows,chipCols), (numChips,4)

       chips       : the list of chips and pixels regions 
                     [[numChannels,chipRows,chipCols], [4], ...]
       pixelRegion : Save the relative pixel locations for each chip 
                     (startRow,startCol,endRow,endCol)
    '''
    from minibatch import makeContiguous
    pixels, regions = makeContiguous(chips)
    if not pixelRegion :
        if log is not None :
            log.debug('Removing the pixelRegion')
        regions = None
    return pixels, regions

def regularGrid(image, chipSize, skipFactor=0, log=None) :
    '''This chips the region into non-overlapping sub-regions. All partial
       border chips will be disgarded from the returned array.

       image       : numpy.ndarray formatted (numChannels, rows, cols)
       chipSize    : Size of chips to be extracted (rows, cols)
       stepFactor  : Number of pixels to advance to next chip start
                     (rows, cols)
       log         : Logger to use
    '''
    return overlapGrid(image, chipSize, log=log,
                       stepFactor=(chipSize[0] * (skipFactor+1),
                                   chipSize[0] * (skipFactor+1)))

def overlapGrid(image, chipSize, stepFactor, log=None) :
    '''This chips the region into non-overlapping sub-regions. All partial
       border chips will be discarded from the returned array.

       image       : numpy.ndarray formatted (numChannels, rows, cols)
       chipSize    : Size of chips to be extracted (rows, cols)
       stepFactor  : Number of pixels to advance to next chip start
                     (rows, cols)
       log         : Logger to use
    '''
    if log is not None :
        log.info('Subdividing the Image')

    # grab an grid of chips
    chips = []
    chipRows, chipCols = chipSize[0], chipSize[1]
    for row in range(0, image.size[1] - chipSize, stepFactor[0]) :
        for col in range(0, image.size[2] - chipSize, stepFactor[1]) :
            chips.append((image[:, row : row + chipRows, col : col + chipCols],
                          [row, col, row + chipRows, col + chipCols]))
    return chips

def randomChip(image, chipSize, numChips=100, log=None) :
    '''This chips the region randomly for a specified number of chips. 

       image       : numpy.ndarray formatted (numChannels, rows, cols)
       chipSize    : Size of chips to be extracted (rows, cols)
       numChips    : Number of chips to extract from the image
       log         : Logger to use
    '''
    from numpy.random import uniform
    if log is not None :
        log.info('Subdividing the Image')

    # randomly generate the chip pairs
    chips = []
    chipRows, chipCols = chipSize[0], chipSize[1]
    randRows = uniform(low=0, high=image.size[1]-chipRows,
                       size=numChips).astype('int32')
    randCols = uniform(low=0, high=image.size[2]-chipCols,
                       size=numChips).astype('int32')
    for row, col in zip(randRows, randCols) :
        chips.append((image[:, row : row + chipRows, col : col + chipCols],
                      [row, col, row + chipRows, col + chipCols]))
    return chips

def selectiveChip(image, chipSize, log=None) :
    raise Exception('Please Implement selectiveChip().')

def applyChipping (images, log=None, *chipFunc, **kwargs) :
    '''A utility to run the chipping function on a number of images

       images   : List of images in memory
       log      : Logger to use
       *chipFunc: Chipping function to use
       **kwargs : Chipping function parameters
    '''
    chips = []
    for im in images :
        chips.extend(*chipFunc(im, **kwargs))
    return chips
