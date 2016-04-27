import numpy as np
import PIL.Image as Image
from nn.datasetUtils import normalize as norm

def saveNormalizedImage(image, path) :
    '''Save the image to a file. This performs'''
    im = Image.fromarray(np.uint8(norm(image) * 255))
    im.save(path)

def saveTiledImage(image, path, imageShape, spacing,
                   tileShape=None, interleave=True) :
    '''Create a tiled image of the weights. 
    '''
    import cv2

    im = image
    if len(image.shape) == 4 :
        im = im.reshape((image.shape[0], image.shape[1], 
                         image.shape[2] * image.shape[3]))
    elif len(image.shape) == 2 :
        im = im.reshape((image.shape[0], 1, image.shape[1]))
    numChannels = im.shape[1]

    # reset if the input doesn't support
    if numChannels != 3 :
        interleave = False

    # calculate an appropriate tiling for the image
    if tileShape is None :
        import math
        colTiles = math.ceil(math.sqrt(im.shape[0]))
        tileShape = (int(math.ceil(im.shape[0] / colTiles)), int(colTiles))

    # create a buffer for the output image
    outputShape = (numChannels,
                   (imageShape[0] + spacing) * tileShape[0] - spacing,
                   (imageShape[1] + spacing) * tileShape[1] - spacing)
    output = np.zeros(outputShape, dtype='uint8')

    # populate the tiles with the pixel data
    for ii in range(tileShape[0]) :
        for jj in range(tileShape[1]) :
            # NOTE: we assume the images are aligned in each row
            imageOffset = ii * tileShape[1] + jj

            # collect each chip for the channels
            chip = norm(im[imageOffset, :, :]) * 255
            chip = chip.reshape((numChannels,
                                 imageShape[0],
                                 imageShape[1]))

            # align into the output buffer -- BGR for openCV
            outLoc = (ii * (imageShape[0] + spacing),
                      jj * (imageShape[1] + spacing))
            if numChannels == 3 :
                output[:, outLoc[0] : outLoc[0] + imageShape[0],
                          outLoc[1] : outLoc[1] + imageShape[1]] = \
                    chip[[2,1,0],:,:]
            else :
                output[:, outLoc[0] : outLoc[0] + imageShape[0],
                          outLoc[1] : outLoc[1] + imageShape[1]] = \
                    chip[:,:,:]

    if interleave :
        # write the image to disk
        cv2.imshow("open", output)
        #cv2.imwrite(path, output)
    else :
        output = output.reshape((outputShape[0] * outputShape[1], 
                                 outputShape[2]))
        im = Image.fromarray(output)
        im.save(path)
