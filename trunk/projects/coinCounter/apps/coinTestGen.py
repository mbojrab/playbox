import argparse
import cv2
import os
import numpy as np

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
    options = parser.parse_args()

    fullPath = options.outDir
    if not os.path.exists(fullPath) :
        os.makedirs(fullPath)

    # setup the video device
    videoCap = cv2.VideoCapture(options.device)

    # start capturing frames
    count = 0
    ret = True
    while ret :
        # capture a frame
        ret, frame = videoCap.read()

        ch = chr(cv2.waitKey(1) & 255)
        if ch== 'l' :
            cv2.imwrite(os.path.join(fullPath, str(count) + '.tiff'), frame)
            count += 1
        if ch== 'q' :
            ret = False
        cv2.imshow('image', frame)

    # cleanup
    videoCap.release()
    cv2.destroyAllWindows()

