import cv2
import logging
import os
import theano
import numpy as np
from PIL import Image, ImageDraw

options = None
windowName = 'Kirtland AF Base'
thumbOrig = None
mouseLocation = None
referenceVector = None
network = None

def selectRegion (event, x, y, flags, param) :
    if event == cv2.EVENT_LBUTTONDBLCLK :
        # record the mouse click
        global mouseLocation, referenceVector
        halfBox = options.chipSize / 2
        mouseLocation = [(x-halfBox, y-halfBox), (x+halfBox, y+halfBox)]

        # chip and find the reference encoded vector
        referenceVector = network.classifyAndSoftmax(
            np.array(thumbOrig.crop(
                (mouseLocation[0][0], mouseLocation[0][1], 
                 mouseLocation[1][0], mouseLocation[1][1]))))
        print referenceVector

def subdivideImage(image, chipSize, stepFactor=1, 
                   batchSize=1, shuffle=False, log=None) :
    import math
    import random
    from nn.datasetUtils import normalize

    if log is not None :
        log.info('Subdividing the Image...')

    # convert to np array
    imageNP = normalize(np.asarray(image.getdata(), 
                                   dtype=theano.config.floatX))
    imageNP = np.resize(imageNP, (1, image.size[0], image.size[1]))

    # setup a grid of chips
    trainingChips = []

    # grab an overlapped grid of chips -- 
    # we are forcing the chips to be square here
    numRows, numCols = chipSize, chipSize
    for ii in range(0, image.size[0] - chipSize, stepFactor) :
        for jj in range(0, image.size[1] - chipSize, stepFactor) :
            trainingChips.append(
                (imageNP[:,ii:ii+numRows,jj:jj+numCols],
                 [ii,jj,ii+numRows,jj+numCols]))

    # randomize the data -- for training purposes
    if shuffle :
        if log is not None :
            log.info('Shuffling the Chips...')
        random.shuffle(trainingChips)

    # create a mini-batch of the data
    numBatches = int(math.floor(float(len(trainingChips)) / float(batchSize)))
    trainingChips = np.concatenate(trainingChips)
    return np.resize(np.concatenate(trainingChips[::2]), 
                     (numBatches,batchSize,1,numRows,numCols)), \
           np.resize(np.concatenate(trainingChips[1::2]),
                     (numBatches,batchSize,4))

def createNetwork(image, log=None) :
    from nn.net import ClassifierNetwork

    # load a previously created network
    if options.synapse is not None :
        if log is not None :
            log.info('Loading Network from Disk...')
        network = ClassifierNetwork(options.synapse, log)

    # create a newly trained network on the specified image
    else :
        import time
        from ae.net import StackedAENetwork
        from nn.datasetUtils import loadShared
        from ae.convolutionalAE import ConvolutionalAutoEncoder
        from ae.contiguousAE import ContractiveAutoEncoder
        from numpy.random import RandomState
        import theano.tensor as t
        from operator import mul

        if log is not None :
            log.info('Training new Network...')

        # create a random number generator for efficiency
        rng = RandomState(int(time.time()))

        # divide the image into chips
        chips, regions = subdivideImage(image, options.chipSize, 
                                        options.chipSize / 2,
                                        options.batchSize, True)

        if log is not None :
            log.info('Intializing the SAE...')

        # create the SAE
        network = StackedAENetwork((loadShared(chips, True), None), log=log)
        input = t.ftensor4('input')

        # add convolutional layers
        network.addLayer(ConvolutionalAutoEncoder(
            layerID='c1', input=input,
            inputSize=chips.shape[1:], 
            kernelSize=(options.kernel,chips.shape[2],5,5),
            downsampleFactor=(2,2), randomNumGen=rng,
            learningRate=options.learnC))
        '''
        network.addLayer(ConvolutionalAutoEncoder(
            layerID='c2',
            input=network.getNetworkOutput(), 
            inputSize=network.getNetworkOutputSize(), 
            kernelSize=(options.kernel,options.kernel,5,5),
            downsampleFactor=(2,2), randomNumGen=rng,
            learningRate=options.learnC))
        '''

        # add fully connected layers
        numInputs = reduce(mul, network.getNetworkOutputSize()[1:])
        network.addLayer(ContractiveAutoEncoder(
            layerID='f3', input=network.getNetworkOutput().flatten(2),
            inputSize=(network.getNetworkOutputSize()[0], numInputs),
            numNeurons=int(options.hidden*1.5),
            learningRate=options.learnF, randomNumGen=rng))
        network.addLayer(ContractiveAutoEncoder(
            layerID='f4', input=network.getNetworkOutput(),
            inputSize=network.getNetworkOutputSize(),
            numNeurons=options.hidden, learningRate=options.learnF,
            randomNumGen=rng))
        network.addLayer(ContractiveAutoEncoder(
            layerID='f5', input=network.getNetworkOutput(),
            inputSize=network.getNetworkOutputSize(),
            numNeurons=options.neuron, learningRate=options.learnF,
            randomNumGen=rng))

        if log is not None :
            log.info('Entering Training...')

        # TODO: this could make for a great demo visual to create a blinking
        #       image of the chips which are currently being activated
        network.trainGreedyLayerwise(options.numEpochs)

        # save trained network -- just in case
        if log is not None :
            log.info('Saving Trained Network...')
        network.save(os.path.basename(options.image).replace(
                        '.png', '_preTrainedSAE_epoch' + \
                        str(options.numEpochs) + '.pkl.gz'))

        # cast to the correct network type
        network.__class__ = ClassifierNetwork

    return network


if __name__ == '__main__' :
    global network

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', dest='logfile', type=str, default=None,
                        help='Specify log output file.')
    parser.add_argument('--level', dest='level', default='INFO', type=str, 
                        help='Log Level.')
    parser.add_argument('--syn', dest='synapse', type=str, default=None,
                        help='Load from a previously saved network.')
    parser.add_argument('--learnC', dest='learnC', type=float, default=.0031,
                        help='Rate of learning on Convolutional Layers.')
    parser.add_argument('--learnF', dest='learnF', type=float, default=.0015,
                        help='Rate of learning on Fully-Connected Layers.')
    parser.add_argument('--contrF', dest='contrF', type=float, default=.01,
                        help='Rate of contraction of the latent space on ' +
                             'Fully-Connected Encoders.')
    parser.add_argument('--chipSize', dest='chipSize', type=int, default=30,
                        help='Size of chip the network should ingest.')
    parser.add_argument('--scale', dest='scale', type=int, default=10,
                        help='Scale down the image for display purposes.')
    parser.add_argument('--kernel', dest='kernel', type=int, default=50,
                        help='Number of Convolutional Kernels in each Layer.')
    parser.add_argument('--hidden', dest='hidden', type=int, default=400,
                        help='Number of Neurons in Hidden Layer.')
    parser.add_argument('--neuron', dest='neuron', type=int, default=250,
                        help='Number of Neurons in Output Layer.')
    parser.add_argument('--epoch', dest='numEpochs', type=int, default=15,
                        help='Number of epochs to run per layer during ' +
                             'unsupervised pre-training.')
    parser.add_argument('--batch', dest='batchSize', type=int, default=1,
                        help='Batch size for training and test sets.')
    parser.add_argument('--base', dest='base', type=str, default='./leNet5',
                        help='Base name of the network output and temp files.')
    parser.add_argument('image', help='Input image to train.')
    options = parser.parse_args()

    # setup mouse input
    cv2.namedWindow(windowName)
    cv2.setMouseCallback(windowName, selectRegion)

    # setup the logger
    log = logging.getLogger('likenessFinder')
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

    # read the file into memory
    if log is not None :
        log.info('Reading the Input into Memory...')
    imageFromDisk = Image.open(options.image)
    imageFromDisk.load()

    # decimate everything for speed
    thumbOrig = imageFromDisk.copy()
    thumbOrig.thumbnail((imageFromDisk.size[0] / options.scale,
                         imageFromDisk.size[1] / options.scale),
                        Image.ANTIALIAS)

    # load a network from disk or train a new network
    network = createNetwork(thumbOrig, log)
    
    print network.classifyAndSoftmax(
        np.resize(np.array(thumbOrig.crop((0, 0, 30, 30))), (1,1,30,30)))

    '''

    # grab the image chips
    chips, regions = subdivideImage(imageFromDisk, options.chipSize, 
                                    options.chipSize / 2, options.batchSize)

    regionsShape = regions.shape
    regions /= options.scale
    regions.astype('int32')
    '''

    # allow user input
    count = 0
    while True :
        global mouseLocation, referenceVector
        thumb = thumbOrig.convert('RGB')
        if mouseLocation is not None :
            draw = ImageDraw.Draw(thumb, mode='RGBA')
            draw.rectangle([mouseLocation[0], mouseLocation[1]], 
                           fill=(0,255,0,100))
        '''
        draw = ImageDraw.Draw(thumb)
        [draw.rectangle([(i[0], i[1]), (i[2], i[3])], outline=(0,150,0)) \
            for i in regions[count % len(regions)]]
        '''
        cv2.imshow(windowName, np.array(thumb))

        ch = chr(cv2.waitKey(1) & 255)
        
        # 'q' drops out of the loop
        if ch== 'q' :
            # cleanup
            cv2.destroyWindow(windowName)
            break
