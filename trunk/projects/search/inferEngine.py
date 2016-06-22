import numpy as np
from mstarTrainingSetGenerator import grabChip, getResearchXML, parseXML
from sarSearch import preProcessing, detectImage

colors = [[[187,38,90],
           [210,100,100],
           [245,45,38],
           [340,1,94],
           [261,11,90]],  
          [[195,80,49],
           [180,77,69],
           [47,61,96],
           [28,68,93],
           [240,30,29]],
          [[166,100,100],
           [40,26,72],
           [199,34,98],
           [28,59,86],
           [270,5,15]]]

def parseImage(image) :
    from dataset.reader import openSICD

    # read the objects from the research XML
    objects = parseXML(getResearchXML(image)).getroot().findall(".//Object")

    # read the wideband data
    wbData, _ = openSICD(options.image)
    return wbData, objects

def classifyImage (network, batchSet, wbData, objects, opts, threads, log, imagePath) :
    #import cv2
    import os
    import PIL.Image as Image
    from PIL import ImageDraw
    from dataset.reader import convertPhaseAmp

    base = os.path.join(os.path.split(__file__)[0], 'webpage', 'static',
                        os.path.basename(imagePath))

    # get the chip size
    chipSize = network.getNetworkInputSize()[2:]

    # extract the chips
    chipDetected, chipLocation, results = [], [], []
    for ii, observation in enumerate(objects) :

        # read the location and extract the chip
        chip, loc = grabChip(wbData, observation, chipSize[0])

        # detect the chip for later
        chipDetected.append(detectImage(chip, opts, threads))
        chipLocation.append([loc[2], loc[0], loc[3], loc[1]])
        Image.fromarray(chipDetected[-1]).save(
            base.replace('SICD.nitf', '_' + str(ii) + '.jpeg'))


        # insert the chip into the pre-allocated buffer
        batchSet[ii % batchSet.shape[0]] = convertPhaseAmp(chip, log)

        # once the buffer is full classify it
        if (ii+1) % batchSet.shape[0] == 0 :
            results.extend(network.classifyAndSoftmax(batchSet))

    # add any partially filled batches
    remainder = len(objects) % batchSet.shape[0]
    if remainder > 0 :
        results.extend(network.classifyAndSoftmax(batchSet)[:remainder])

    # detect the full image
    fullDetected = Image.fromarray(detectImage(wbData, opts, threads))
    width = 500
    factor = float(width)/fullDetected.size[0]
    fullDetected.thumbnail((width, int(fullDetected.size[1] * factor)),
                           Image.ANTIALIAS)
    fullDetected = fullDetected.convert('RGB')

    # apply the locations to the overview image
    draw = ImageDraw.Draw(fullDetected)
    for label, loc in zip(network.convertToLabels(results[0]), chipLocation) :
        # draw a box around the object
        if label.upper() != "MISC" :
            draw.rectangle([int(loc[0]*factor), int(loc[1]*factor),
                            int(loc[2]*factor), int(loc[3]*factor)],
                           fill=tuple(labelColors[label] + [5]))

    # return the results
    fullDetected.save(base.replace('SICD.nitf', '_full.jpeg'))
    return (fullDetected, np.concatenate(chipDetected), results)


if __name__ == '__main__' :
    import argparse
    import theano
    from nn.net import ClassifierNetwork as Net
    from nn.profiler import setupLogging

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', dest='confDir', type=str, default=None,
                        help='Specify the conf/ directory.')
    parser.add_argument('--log', dest='logfile', type=str, default=None,
                        help='Specify log output file.')
    parser.add_argument('--level', dest='level', default='INFO', type=str, 
                        help='Log Level.')
    parser.add_argument('--syn', dest='synapse', type=str, default=None,
                        help='Load from a previously saved network.')
    parser.add_argument('--color', dest='color', type=int, default=0,
                        help='Load a particular color palette.')
    parser.add_argument('image', help='Input image to classify.')
    options = parser.parse_args()

    # setup the logger
    log = setupLogging('MSTARInference', options.level, options.logfile)

    if options.synapse is None :
        raise Exception('The user must specify a synapse file via --syn')
    network = Net(filepath=options.synapse, log=log)

    # allocate memory for our batch size --
    # this will allow fast classification
    batchSet = np.zeros(network.getNetworkInputSize(), 
                        dtype=theano.config.floatX)

    # setup for processing
    opts, _, threads = preProcessing(options.confDir)

    labelColors = {}
    for ii, label in enumerate(network.getNetworkLabels()) :
        labelColors[label] = colors[options.color][ii]

    # process the image(s)
    wbData, objects = parseImage(options.image)
    classifyImage(network, batchSet, wbData, objects, opts, threads, log, options.image)
