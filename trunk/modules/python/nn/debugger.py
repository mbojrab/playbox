import numpy as np
from nn.datasetUtils import norm

def saveNormalizedImage(image, path) :
    import PIL.Image as Image
    im = Image.fromarray(np.uint8(norm(image) * 255))
    im.save(path)

def saveTiledImage(image, path, imageShape, spacing,
                   tileShape=None, interleave=True) :
    '''Create a tiled image of the weights. 
    '''
    import cv2
    im = image
    if len(image.shape) == 2 :
        im = im.reshape((1, image.shape[0], image.shape[1]))
    numChannels = image.shape[0]

    # calculate an appropriate tiling for the image
    imNumPixels = imageShape[0] * imageShape[1]
    if tileShape is None :
        import math
        totNumPixels = im.shape[0] * im.shape[1]
        numTiles = math.ceil(totNumPixels / imNumPixels)
        colTiles = math.ceil(math.sqrt(numTiles))
        tileShape = (math.ceil(numTiles / colTiles), colTiles)
    print "tileShape: " + str(tileShape)

    # create a buffer for the output image
    outputShape = (numChannels,
                   imageShape[0] + spacing * tileShape[0] - spacing,
                   imageShape[1] + spacing * tileShape[1] - spacing)
    output = np.zeros(outputShape, dtype='uint8')
    print "outputShape: " + str(outputShape)

    # populate the tiles with the pixel data
    for ii in range(tileShape[0]) :
        for jj in range(tileShape[1]) :
            # NOTE: we assume the images are aligned in each row
            imageOffset = ii * tileShape[1] + jj
            if imageOffset < im.shape[1] :

                # collect each chip for the channels
                chip = norm(im[:, imageOffset : imageOffset + imNumPixels])
                chip = chip.reshape(imageShape) * 255

                # align into the output buffer -- BGR for openCV
                outLoc = (ii * (imageShape[0] + spacing),
                          jj * (imageShape[1] + spacing[1]))
                output[:, outLoc[0] : outLoc[0] + imageShape[0],
                          outLoc[1] : outLoc[1] + imageShape[1]] = \
                      chip[[2,1,0],:,:]

    # write the image to disk
    cv2.imshow("open", output)
    #cv2.imwrite(path, output)
    return output
