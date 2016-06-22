import numpy as np

def preProcessing (confDir) :
    import multiprocessing
    from pynpar import npar
    import pynpar.apodize as apodize
    import pynpar.rsa as rsa

    threads = multiprocessing.cpu_count()
    detectedOptions = npar.DetectedOptions(confDir, threads, False, 
                                           apodize.HNNS_1D, 
                                           rsa.RSA3, 
                                           "GDM", 3, True, True);
    return (detectedOptions, 
            npar.makeSharedLogger(), 
            threads)

def detectImage (sarImage, detectedOptions, threads) :
    from pynpar import npar
    from coda import algs_remap

    # detect the complex image
    detected = npar.detect(sarImage, threads)

    # grab the stats
    mask = np.ones_like(detected).astype(bool)
    chMedian, chMean, chSkewness = algs_remap.getRsa3Stats(detected, mask, 
                                                           threads)
    (cl, ch) = algs_remap.calcCLCH(algs_remap.RSA3Coefficients(), 
                                   chMedian, chMean, chSkewness, 0., 0.)
    detected = npar.spectralShape(detected, cl, ch,
                                  detectedOptions.getApodizationType(), 
                                  threads)

    # remap the buffer
    detected = algs_remap.remapGDM(detected, cl, ch, threads)

    # remap to PEDF
    return algs_remap.remapPEDF(detected, threads)

def selectiveSearch (imagePath, confDir) :
    import cv2
    import PIL.Image as Image
    from dataset.reader import openSICD, openSIO

    # setup for processing
    opts, log, threads = preProcessing(confDir)

    # read the sicd and convert to bytes
    if imagePath.lower().endswith('.sio') :
        wbData = openSIO(imagePath)
    else :
        wbData, _ = openSICD(imagePath)
    detected = detectImage(wbData, opts, threads)

    im = Image.fromarray(detected)

    windowName = "DetectedImage"
    while True :
        cv2.imshow(windowName, np.array(im))#detected)

        ch = chr(cv2.waitKey(1) & 255)

        # 'q' drops out of the loop
        if ch== 'q' :
            # cleanup
            cv2.destroyWindow(windowName)
            break

if __name__ == '__main__' :
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', dest='confDir', type=str, default=None,
                        help='Specify the conf/ directory.')
    parser.add_argument('image', help='Input image to classify.')
    options = parser.parse_args()

    selectiveSearch (options.image, options.confDir)
