import argparse
import cv2
import os
import logging
import numpy as np
from nn.net import ClassifierNetwork as Net

def findCoins(grey) :
    # use hough transform to identify circular objects
    circles = cv2.HoughCircles(grey, cv2.cv.CV_HOUGH_GRADIENT, 
                               1, grey.shape[0] / 16, 
                               param1=475, param2=15,
                               minRadius=5, maxRadius=100)
    # round to the nearest pixel
    if circles is not None :
        circles = np.uint16(np.around(circles))
    return circles

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', dest='logfile', type=str, default=None,
                        help='Specify log output file.')
    parser.add_argument('--level', dest='level', default='INFO', type=str, 
                        help='Log Level.')
    parser.add_argument('--input', dest='input', type=str,
                        help='Specify an input image to classify.')
    parser.add_argument('--out', dest='outDir', type=str, default='./coins',
                        help='Specify an output directory to place the ' +
                        'training dataset.')
    parser.add_argument('--device', dest='device', type=int, default=0,
                        help='Specify the index of the video device to use.')
    parser.add_argument('--syn', dest='synapse', type=str, default=None,
                        help='Load from a previously saved network.')
    options = parser.parse_args()

    # setup the logger
    log = logging.getLogger('coinCounter')
    log.setLevel(options.level.upper())
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    stream = logging.StreamHandler()
    stream.setLevel(options.level.upper())
    stream.setFormatter(formatter)
    log.addHandler(stream)
    if options.logfile is not None :
        logFile = logging.FileHandler(options.logfile)
        logFile.setLevel(options.level.upper())
        logFile.setFormatter(formatter)
        log.addHandler(logFile)

    if options.synapse is None :
        raise Exception('The user must specify a synapse file via --syn')
    network = Net(filepath=options.synapse, log=log)

    # input can come from a file or an video streams
    if options.input is not None :
        while True :
            from nn.datasetUtils import readImage
            #from PIL import Image
            #im = np.array(Image.open(options.input))
            im = readImage(options.input, log=log)

            #circles = findCoins(im[:, :, 0])
            circles = findCoins(im[0, :, :])

            # draw boxes around any coins identified
            #cv2.imshow('original', im[:,:,::-1])

            chips = []
            if circles is not None :
                for c in circles[0,:]:
                    radius = int(c[2] * 1.2)
                    upperLeft = (c[0]-radius, c[1]-radius)
                    lowerRight = (c[0]+radius, c[1]+radius)

                    # chip the coin
                    chip = im[upperLeft[1]:lowerRight[1], 
                              upperLeft[0]:lowerRight[0],:]

                    # resize chip to fit the network input dimensions
                    chipDims = chip.shape
                    netInputDims = network.getNetworkInputSize()[1:]
                    print chipDims
                    print netInputDims
                    
                    if chip.shape != netInputDims :
                        from scipy.misc import imresize
                        chip = imresize(chip, netInputDims, 
                                        interp='bicubic', mode=None)                            
                    print chip.shape

                    print network.classify(chip)
                    cv2.imwrite('chip.tiff', chip[:,:,::-1])

                    # draw a box on the image to indicate we found the coin
                    cv2.rectangle(im, upperLeft, lowerRight, (0,255,0), 2)

                #cv2.imshow('coins found', im[:,:,::-1])

            # check the user input
            ch = chr(cv2.waitKey(1) & 255)
            if ch== 'q' :
                # cleanup
                cv2.destroyAllWindows()
                break

    else :

        # create the output directory if it doesn't exist
        if not os.path.exists(options.outDir) :
            os.makedirs(options.outDir)

        # setup the video device
        videoCap = cv2.VideoCapture(options.device)

        # start capturing frames
        count = 0
        while True :
            # capture a frame
            ret, frame = videoCap.read()

            ch = chr(cv2.waitKey(1) & 255)
            if ch== 'l' :
                cv2.imwrite(os.path.join(options.outDir,
                                         str(count) + '.tiff'), frame)
                count += 1
            if ch== 'q' :
                # cleanup
                videoCap.release()
                cv2.destroyAllWindows()
                break
            cv2.imshow('image', frame)


