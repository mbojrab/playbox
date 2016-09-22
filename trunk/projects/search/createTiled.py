import os
import numpy as np
from sarSearch import detectImage

def processDirectory(dir, opts, log, threads) :
    import glob
    from dataset.reader import openSIO
    from dataset.debugger import saveTiledImage

    # grep for sio's
    sios = glob.glob(os.path.join(dir, "*.sio"))

    print("Found [" + str(len(sios)) + "] sio files in [" +
          os.path.basename(dir) + "]")

    imageTiles = []
    for ii, sio in enumerate(sios) :
        
        # read the sicd and convert to bytes
        if sio.lower().endswith('.sio') :
            wbData = openSIO(sio)

        if ii == 0 :
            imageShape = wbData.shape

        detected = detectImage(wbData, opts, threads)
        imageTiles.append(np.reshape(detected, 
            (1, imageShape[0], imageShape[1])))

    # reshape
    imageTiles = np.reshape(np.concatenate(imageTiles), 
                            (len(sios), 1, imageShape[0], imageShape[1]))
    dirSplit = os.path.split(dir)
    saveTiledImage(image=imageTiles,
                   path=os.path.join(dirSplit[0], dirSplit[1] + '.png'),
                   imageShape=(imageShape[0], imageShape[1]),
                   spacing=2, interleave=True)
        
if __name__ == '__main__' :
    import argparse
    from concurrent.futures import ThreadPoolExecutor
    from sarSearch import preProcessing

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', dest='confDir', type=str, default=None,
                        help='Specify the conf/ directory.')
    parser.add_argument('dir', help='Input directory to create tiled image.')
    options = parser.parse_args()

    # setup for processing
    opts, log, threads = preProcessing(options.confDir)

    with ThreadPoolExecutor(max_workers=4) as e :
        for dir in os.listdir(options.dir) :
            dir = os.path.join(options.dir, dir)
            if os.path.isdir(dir) :
                print("Processing Directory [" + dir + "]")
                e.submit(processDirectory, dir, opts, log, threads)
