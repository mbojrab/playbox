def grid(image, chipSize, skipFactor=0, pixelRegion=False, log=None):
    '''This chips the region into non-overlapping sub-regions. All partial
       border chips will be disgarded from the returned array.

       image       : numpy.ndarray formatted (numChannels, rows, cols)
       chipSize    : Size of chips to be extracted (rows, cols)
       stepFactor  : Number of pixels to advance to next chip start
                     (rows, cols)
       pixelRegion : Save the relative pixel locations for each chip 
                     (startRow,startCol,endRow,endCol)
       log         : Logger to use
    '''
    return overlapGrid(image, chipSize, log=log, pixelRegion=pixelRegion,
                       stepFactor=(chipSize[0] * (skipFactor+1),
                                   chipSize[0] * (skipFactor+1)))

def overlapGrid(image, chipSize, stepFactor, 
                pixelRegion=False, log=None) :
    '''This chips the region into non-overlapping sub-regions. All partial
       border chips will be discarded from the returned array.

       image       : numpy.ndarray formatted (numChannels, rows, cols)
       chipSize    : Size of chips to be extracted (rows, cols)
       stepFactor  : Number of pixels to advance to next chip start
                     (rows, cols)
       pixelRegion : Save the relative pixel locations for each chip 
                     (startRow,startCol,endRow,endCol)
       log         : Logger to use
    '''
    from minibatch import makeContiguous
    if log is not None :
        log.info('Subdividing the Image')

    # grab an grid of chips
    chips = []
    chipRows, chipCols = chipSize[0], chipSize[1]
    for row in range(0, image.size[1] - chipSize, stepFactor[0]) :
        for col in range(0, image.size[2] - chipSize, stepFactor[1]) :
            chips.append((image[:, row : row + chipRows, col : col + chipCols],
                          [row, col, row + chipRows, col + chipCols]))

    # make data contigous in memory --
    # all pixels will be arranged into a single tensor of size 
    # (numChips,numChannels,chipRows,chipCols), (numChips,4)
    pixels, regions = makeContiguous(chips)
    if not pixelRegion :
        regions = None

    return pixels, regions
