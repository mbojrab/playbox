import numpy as np
import PIL.Image as Image
from dataset.reader import normalize as norm

def saveNormalizedImage(image, path) :
    '''Save the image to a file. This performs a naive normalization of the 
       array and outputs a grey scale image. The input is assumed to be float
       data.

       image: 2D tensor (rows, cols)
       path:  Output path to image
    '''
    Image.fromarray(np.uint8(norm(image) * 255)).save(path)

def saveTiledImage(image, path, imageShape, spacing=2,
                   tileShape=None, interleave=True) :
    '''Create a tiled image of the weights 

       image     : Tensor to convert to a tiled image
                   2D tensor (numKernels, inputSize)
                   3D tensor (numKernels, numChannels, inputSize)
                   4D tensor (numKernels, numChannels, numRows, numCols)
       path      : Output path to image
       imageShape: Shape of each tiled image in output
       spacing   : Size of pixel border around each tileShape
       tileShape : User-specified tiling pattern
       interleave: Create a colorize output image from channels
                   False will separate and stack the image channels
    '''
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

            if imageOffset < im.shape[0] :
                # collect each chip for the channels
                chip = norm(im[imageOffset, :, :]) * 255
                chip = chip.reshape((numChannels,
                                     imageShape[0],
                                     imageShape[1]))

                # align into the output buffer -- BGR for openCV
                outLoc = (ii * (imageShape[0] + spacing),
                          jj * (imageShape[1] + spacing))
                output[:, outLoc[0] : outLoc[0] + imageShape[0],
                          outLoc[1] : outLoc[1] + imageShape[1]] = chip[:,:,:]

    # write the image to disk
    if interleave :
        Image.merge("RGB", (Image.fromarray(output[0]),
                            Image.fromarray(output[1]),
                            Image.fromarray(output[2]))).save(path)
    else :
        # PIL will write the output in its current (stacked) format
        output = output.reshape((outputShape[0] * outputShape[1], 
                                 outputShape[2]))
        Image.fromarray(output).save(path)
