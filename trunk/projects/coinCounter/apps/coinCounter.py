import cv2
import os
import logging
import numpy as np
#from nn.net import ClassifierNetwork as Net
from nn.net import ClassifierNetwork as Net
from nn.datasetUtils import convertImageToNumpy
from PIL import Image, ImageDraw
import theano

coinValue = [.10, .0, .05, .01, .25]
#coinValue = [.05, .01, .25]

def findCoins(grey) :
    # use hough transform to identify circular objects
    circles = cv2.HoughCircles(grey, cv2.cv.CV_HOUGH_GRADIENT, 
                               1, grey.shape[0] / 16, 
                               param1=475, param2=25,
                               minRadius=5, maxRadius=100)
    '''
    circles = cv2.HoughCircles(grey, cv2.cv.CV_HOUGH_GRADIENT, 
                               1, grey.shape[0] / 16, 
                               param1=500, param2=16,
                               minRadius=3, maxRadius=60)
    '''
    # round to the nearest pixel
    if circles is not None :
        circles = np.uint16(np.around(circles))
    return circles

def getChipArea(circle) :
    '''return (radius, upperLeft, lowerRight)'''
    radius = int(circle[2] * 1.2)
    return (radius, (circle[0]-radius, circle[1]-radius),
                    (circle[0]+radius, circle[1]+radius))

def countCoins(image, batchSet, show=False, debug=False) :
    # find likely areas for coins to exist
    circles = findCoins(np.array(image)[:, :, 0])

    # run the coins through the network to get their value
    ii = 0
    totalValue = 0.
    if circles is not None :
        for c in circles[0,:] :
            # chip the coin
            radius, upperLeft, lowerRight = getChipArea(c)
            chip = image.crop((upperLeft[0], upperLeft[1], 
                               lowerRight[0], lowerRight[1]))

            # resize chip to fit the network input dimensions
            chipDims = chip.size
            netInputDims = network.getNetworkInputSize()[2:]

            # if its too small interpolate
            if chipDims[0] < netInputDims[0] or \
               chipDims[1] < netInputDims[1] :
                chip = chip.resize(netInputDims, Image.BICUBIC)
            # if its too large downsample
            elif chipDims[0] > netInputDims[0] or \
                 chipDims[1] > netInputDims[1] :
                chip.thumbnail(netInputDims, Image.ANTIALIAS)

            # insert the chip into the pre-allocated buffer
            chipPosition = ii % batchSet.shape[0]
            batchSet[chipPosition] = convertImageToNumpy(chip, True)
            if debug :
                cv2.imwrite('chipNP_' + str(ii) + '.tiff', batchSet[chipPosition,0,:,:]*255.)

            # once the buffer is full classify it
            if (ii+1) % batchSet.shape[0] == 0 :
                result = network.classify(batchSet)
                totalValue += sum([coinValue[cl] for cl in result])
                if debug :
                    print [network.getNetworkLabels()[cl] for cl in result]
                    print result

            if debug :
                chip.save('chip_' + str(ii) + '.tiff')
            ii += 1

        remainder = len(circles[0]) % batchSet.shape[0]
        if remainder > 0:
            result = network.classify(batchSet)[:remainder]
            totalValue += sum([coinValue[cl] for cl in result])
            if debug :
                print [network.getNetworkLabels()[cl] for cl in result]
                print result



    if show :
        # for diagnostics -- draw boxes and display the results
        draw = ImageDraw.Draw(image)
        if circles is not None :
            for c in circles[0,:] :
                # draw a box on the image to indicate we found the coin
                radius, upperLeft, lowerRight = getChipArea(c)
                draw.rectangle([upperLeft, lowerRight], outline=(0,255,0))
            cv2.imshow('update', 
                       cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

    if debug :
        print "Total Value: $" + str(totalValue)

    return totalValue

if __name__ == '__main__' :
    import argparse
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

    # allocate memory for our batch size --
    # this will allow fast classification
    batchSet = np.zeros(network.getNetworkInputSize(), 
                        dtype=theano.config.floatX)

    # input can come from a file or an video streams
    if options.input is not None :
        # read the file into memory
        image = Image.open(options.input)
        image.load()
        countCoins(image, batchSet, True)

    else :
        from scipy.stats import mode
        # create the output directory if it doesn't exist
        if not os.path.exists(options.outDir) :
            os.makedirs(options.outDir)

        # setup the video device
        videoCap = cv2.VideoCapture(options.device)
        videoCap.set(cv2.cv.CV_CAP_PROP_EXPOSURE, .05) 

        # start capturing frames
        ii = 0
        captureMem = 15
        totalValues = [0] * captureMem
        while True :
            # capture a frame
            ret, frame = videoCap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            totalValues[ii%captureMem] = countCoins(Image.fromarray(frame), 
                                                    batchSet)
            ii += 1
            print "Total Value: $" + str(mode(totalValues)[0][0])

            ch = chr(cv2.waitKey(1) & 255)
            if ch== 'r' :
                #import Image
                #image = Image.fromarray(frame)
                totalValues = []
            if ch== 'q' :
                # cleanup
                videoCap.release()
                cv2.destroyAllWindows()
                break


