import argparse
import cv2
import os
import numpy as np

def getChannelCircles(params, frame) :
    height, width, numChannels = frame.shape

    '''
    mask = np.zeros((width,height),np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    cv2.grabCut(frame, mask, (50, 50, width-50, height-50),
                bgdModel, fgdModel, 6, mode=cv2.GC_INIT_WITH_RECT)
    '''
    chan = frame[:,:,params['chan']]
    chan = cv2.medianBlur(chan, 7)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    chan = clahe.apply(chan)
#    ret, chan = cv2.threshold(chan, params['thresh'], 
#                              maxval=255, type=cv2.THRESH_BINARY_INV)
    cv2.imshow('grey', chan)
    return cv2.HoughCircles(chan, cv2.cv.CV_HOUGH_GRADIENT, 
                            params['factor1'], height / params['factor2'],
                            param1=params['param1'], param2=params['param2'],
                            minRadius=0, maxRadius=0)

def saveNextAvailable(params, image):
    # keep searching for an image name that doesn't exist
    imagePath = os.path.join(
        params['fullPath'], str(params['imageCounter']) + '.tiff')
    while os.path.exists(imagePath) :
        params['imageCounter'] += 1
        imagePath = os.path.join(
            params['fullPath'], str(params['imageCounter']) + '.tiff')

    # write the image to this location --
    # we do this because we will periodically delete and recreate different
    # imagery for the training dataset.
    print 'Writing image [' + imagePath + ']'
    cv2.imwrite(imagePath, image)

def userInput(params, frame, circles) :
    # check for user input
    ch = chr(cv2.waitKey(1) & 255)
    if ch== 'q' :
        return False
    elif ch == 'w' :
        params['param1']+=5
        print 'param1: ' + str(params['param1'])
    elif ch == 's' :
        params['param1'] = max(5, params['param1']-5)
        print 'param1: ' + str(params['param1'])
    elif ch == 'd' :
        params['param2']+=5
        print 'param2: ' + str(params['param2'])
    elif ch == 'a' :
        params['param2'] = max(5, params['param2']-5)
        print 'param2: ' + str(params['param2'])
    elif ch == 'r' :
        params['factor1']+=1
        print 'factor1: ' + str(params['factor1'])
    elif ch == 'f' :
        params['factor1'] = max(1, params['factor1']-1)
        print 'factor1: ' + str(params['factor1'])
    elif ch == 't' :
        params['factor2']+=1
        print 'factor2: ' + str(params['factor2'])
    elif ch == 'g' :
        params['factor2'] = max(1, params['factor2']-1)
        print 'factor2: ' + str(params['factor2'])
    elif ch == 'y' :
        params['thresh']+=5
        print 'thresh: ' + str(params['thresh'])
    elif ch == 'h' :
        params['thresh'] = max(0, params['thresh']-5)
        print 'thresh: ' + str(params['thresh'])
    elif ch == 'k' :
        params['chan'] = (params['chan']+1) % 3
        print 'chan: ' + str(params['chan'])
    elif ch == 'l' :
        if circles is not None :
            for c in circles[0,:]:
                radius = int(c[2] * 1.2)
                saveNextAvailable(params, frame[c[1]-radius:c[1]+radius, 
                                                c[0]-radius:c[0]+radius,:])
    return True

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', dest='logfile', type=str, default=None,
                        help='Specify log output file.')
    parser.add_argument('--level', dest='level', default='INFO', type=str, 
                        help='Log Level.')
    parser.add_argument('--out', dest='outDir', type=str, default='./coins',
                        help='Specify an output directory to place the ' +
                        'training dataset.')
    parser.add_argument('--origin', dest='origin', type=str, default='US',
                        help='Country of origin for the currency in this run.')
    parser.add_argument('--type', dest='type', type=str, 
                        help='Type of coin depicted in scene.')
    parser.add_argument('--device', dest='device', type=int, default=0,
                        help='Specify the index of the video device to use.')
    options = parser.parse_args()

    fullPath = os.path.join(options.outDir, options.origin, options.type)
    if not os.path.exists(fullPath) :
        os.makedirs(fullPath)

    # setup the video device
    videoCap = cv2.VideoCapture(options.device)

    # start capturing frames
    ret = True
    params = {'fullPath' : fullPath, 'imageCounter' : 0,
              'param1' : 475, 'param2' : 15, 'thresh' : 125,
              'factor1' : 1, 'factor2' : 16, 'chan' : 2} 
    while ret :

        # capture a frame
        ret, frame = videoCap.read()

        # run a circular hough transform
        circles = getChannelCircles(params, frame)

        if circles is not None :
            circles = np.uint16(np.around(circles))
        ret = userInput(params, frame, circles)

        # draw each circle found onto the image
        if circles is not None :

            # round to the nearest pixel
            for c in circles[0,:]:
                radius = int(c[2] * 1.2)
                # draw the bounding box
                cv2.rectangle(frame, (c[0]-radius, c[1]-radius), 
                              (c[0]+radius, c[1]+radius), (0,255,0), 2)

        # display the resulting frame
        cv2.imshow('frame',frame)

    # cleanup
    videoCap.release()
    cv2.destroyAllWindows()

