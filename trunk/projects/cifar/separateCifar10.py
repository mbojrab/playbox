import os
from cifar10 import load_data
import PIL.Image as Image

if __name__ == '__main__' :
    outDir = './CIFAR-10/'
    train, test, labels = load_data()
    labels = labels['label_names']

    def writeData (outDir, dir, data, labels) :
        # setup the train directories
        subDir = os.path.join(outDir, dir)
        if not os.path.exists(subDir) :
            os.makedirs(subDir)

        # write the label directories
        counts = [0] * len(labels)
        for im, ii in zip(data[0], data[1]) :
            # create the directory
            labelDir = os.path.join(subDir, str(labels[ii]))
            if not os.path.exists(labelDir) :
                os.makedirs(labelDir)

            # write the image to the directory
            imPath = os.path.join(labelDir, str(counts[ii]) + '.tif')
            Image.merge('RGB', (Image.fromarray(im[0]),
                                Image.fromarray(im[1]),
                                Image.fromarray(im[2]))).save(imPath)
            counts[ii] += 1

    # setup the train directories
    writeData (outDir, 'train', train, labels)
        
    # setup the test directories
    writeData (outDir, 'test', test, labels)
