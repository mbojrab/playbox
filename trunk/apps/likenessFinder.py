import cv2
import logging
import os
import theano
import numpy as np
from PIL import Image, ImageDraw, ImageFont

options = None
windowName = 'Kirtland AF Base'
thumbOrig = None
refLocation = None
misLocation = None
referenceVector = None
network = None
matchSelect = False
likeness = None

def convertImageToNP(image) :
    from nn.datasetUtils import normalize

    # convert to np array
    imageNP = normalize(np.asarray(image.getdata(), 
                                   dtype=theano.config.floatX))
    return np.resize(imageNP, (1, image.size[0], image.size[1]))

def selectRegion (event, x, y, flags, param) :
    if event == cv2.EVENT_LBUTTONDBLCLK :
        # record the mouse click
        global refLocation, misLocation, referenceVector, matchSelect, likeness
        halfBox = options.chipSize / 2

        # chip and find the reference encoded vector
        if matchSelect == False :
            refLocation = [(x-halfBox, y-halfBox), (x+halfBox, y+halfBox)]
            referenceVector = network.classifyAndSoftmax(
                np.resize(np.array(thumbOrig.crop(
                    (refLocation[0][0], refLocation[0][1], 
                     refLocation[1][0], refLocation[1][1]))),
                    (1, options.chipSize*options.chipSize)))[1][0]
        else :
            misLocation = [(x-halfBox, y-halfBox), (x+halfBox, y+halfBox)]
            matchVector = network.classifyAndSoftmax(
                np.resize(np.array(thumbOrig.crop(
                    (misLocation[0][0], misLocation[0][1], 
                     misLocation[1][0], misLocation[1][1]))),
                    (1, options.chipSize*options.chipSize)))[1][0]
            if refLocation == None :
                print "Please select a reference region!"
            else :
                likeness = np.dot(referenceVector, matchVector)

def subdivideImage(image, chipSize, stepFactor=1, 
                   batchSize=1, shuffle=False, log=None) :
    import math
    import random

    if log is not None :
        log.info('Subdividing the Image...')

    # convert to np array
    imageNP = convertImageToNP(image)

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

    # divide the image into chips
    chips, regions = subdivideImage(image, options.chipSize, 
                                    options.chipSize / 2,
                                    options.batchSize, True)
    chips = np.resize(chips, 
                      (chips.shape[0], chips.shape[1], 
                       chips.shape[2]*chips.shape[3]*chips.shape[4]))

    # load a previously created network
    if options.synapse is not None :
        if log is not None :
            log.info('Loading Network from Disk...')
        network = ClassifierNetwork(options.synapse, log)

        '''
        if log is not None :
            log.info('Entering Training...')

        # TODO: this could make for a great demo visual to create a blinking
        #       image of the chips which are currently being activated
        globalEpoch = 0
        for layerIndex in range(1,network.getNumLayers()) :
            #network.writeWeights(layerIndex, 00)
            for ii in range(100) :
                #globalEpoch, globalCost = network.trainEpoch(
                #    layerIndex, globalEpoch, options.numEpochs)
                globCost = []
                for localEpoch in range(options.numEpochs) :
                    layerEpochStr = 'Layer[' + str(layerIndex) + '] Epoch[' + \
                                    str(globalEpoch + localEpoch) + ']'
                    print 'Running ' + layerEpochStr
                    locCost = []
                    for ii in range(chips.shape[0]) :
                        locCost.append(network.train(layerIndex, ii))

                    locCost = np.mean(locCost, axis=0)
                    if isinstance(locCost, tuple) :
                        print layerEpochStr + ' Cost: ' + \
                              str(locCost[0]) + ' - Jacob: ' + \
                              str(locCost[1])
                    else :
                        print layerEpochStr + ' Cost: ' + str(locCost)
                    globCost.append(locCost)

                #network.writeWeights(layerIndex, globalEpoch + localEpoch)
                globalEpoch = globalEpoch + options.numEpochs
                network.save('kirtland_afb_neurons500_layer' + \
                             str(layerIndex) + '_epoch' + str(globalEpoch) + \
                             '.pkl.gz')
        '''

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

        if log is not None :
            log.info('Intializing the SAE...')

        # create the SAE
        network = StackedAENetwork((loadShared(chips, True), None), log=log)
        input = t.fmatrix('input')

        '''
        # add convolutional layers
        network.addLayer(ConvolutionalAutoEncoder(
            layerID='c1', input=input,
            inputSize=chips.shape[1:], 
            kernelSize=(options.kernel,chips.shape[2],5,5),
            downsampleFactor=(2,2), randomNumGen=rng,
            learningRate=options.learnC))
        network.addLayer(ConvolutionalAutoEncoder(
            layerID='c2',
            input=network.getNetworkOutput(), 
            inputSize=network.getNetworkOutputSize(), 
            kernelSize=(options.kernel,options.kernel,5,5),
            downsampleFactor=(2,2), randomNumGen=rng,
            learningRate=options.learnC))

        # add fully connected layers
        numInputs = reduce(mul, network.getNetworkOutputSize()[1:])
        network.addLayer(ContractiveAutoEncoder(
            layerID='f3', input=network.getNetworkOutput().flatten(2),
            inputSize=(network.getNetworkOutputSize()[0], numInputs),
            numNeurons=int(options.hidden*1.5),
            learningRate=options.learnF, randomNumGen=rng))
        '''
        network.addLayer(ContractiveAutoEncoder(
            layerID='f1', input=input,
            inputSize=(chips.shape[1], reduce(mul, chips.shape[2:])),
            numNeurons=500, learningRate=options.learnF,
            contractionRate=options.contrF, randomNumGen=rng))
        network.addLayer(ContractiveAutoEncoder(
            layerID='f2', input=network.getNetworkOutput(),
            inputSize=network.getNetworkOutputSize(),
            numNeurons=200, learningRate=options.learnF,
            contractionRate=options.contrF, randomNumGen=rng))
        network.addLayer(ContractiveAutoEncoder(
            layerID='f3', input=network.getNetworkOutput(),
            inputSize=network.getNetworkOutputSize(),
            numNeurons=100, learningRate=options.learnF,
            contractionRate=options.contrF, randomNumGen=rng))
        network.addLayer(ContractiveAutoEncoder(
            layerID='f4', input=network.getNetworkOutput(),
            inputSize=network.getNetworkOutputSize(),
            numNeurons=50, learningRate=options.learnF,
            contractionRate=options.contrF, randomNumGen=rng))

        if log is not None :
            log.info('Entering Training...')

        #network.writeWeights(0, -1)
        # TODO: this could make for a great demo visual to create a blinking
        #       image of the chips which are currently being activated
        globalEpoch = 0
        for layerIndex in range(network.getNumLayers()) :
            for ii in range(4) :
                '''
                globalEpoch, globalCost = network.trainEpoch(
                    layerIndex, globalEpoch, options.numEpochs)
                '''
                globCost = []
                for localEpoch in range(options.numEpochs) :
                    layerEpochStr = 'Layer[' + str(layerIndex) + '] Epoch[' + \
                                    str(globalEpoch + localEpoch) + ']'
                    print 'Running ' + layerEpochStr
                    locCost = []
                    for ii in range(chips.shape[0]) :
                        locCost.append(network.train(layerIndex, ii))

                    locCost = np.mean(locCost, axis=0)
                    if isinstance(locCost, tuple) :
                        print layerEpochStr + ' Cost: ' + \
                              str(locCost[0]) + ' - Jacob: ' + \
                              str(locCost[1])
                    else :
                        print layerEpochStr + ' Cost: ' + str(locCost)
                    globCost.append(locCost)

                    #network.writeWeights(layerIndex, globalEpoch + localEpoch)
                globalEpoch = globalEpoch + options.numEpochs
                network.save('kirtland_afb_neurons500_layer' + \
                             str(layerIndex) + '_epoch' + str(globalEpoch) + \
                             '.pkl.gz')
        #network.trainGreedyLayerwise(options.numEpochs)

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
    global options, thumbOrig, network, matchSelect

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

    '''
    # grab the image chips
    chips, regions = subdivideImage(imageFromDisk, options.chipSize, 
                                    options.chipSize / 2, options.batchSize)
    regionsShape = regions.shape
    regions /= options.scale
    regions.astype('int32')
    '''

    # setup mouse input
    cv2.imshow(windowName, np.array(thumbOrig))
    cv2.setMouseCallback(windowName, selectRegion)

    # allow user input
    count = 0
    while True :
        thumb = thumbOrig.convert('RGB')

        draw = ImageDraw.Draw(thumb, mode='RGBA')
        if refLocation is not None :
            draw.rectangle([refLocation[0], refLocation[1]], 
                           fill=(0,255,0,100))
        if misLocation is not None :
            draw.rectangle([misLocation[0], misLocation[1]], 
                           fill=(0,255,255,100))
        if likeness != None :
            font = ImageFont.truetype("arial.ttf", 50)
            draw.text([20, 850], "Likeness: " + str(likeness),
                      font=font, fill=(150,0,200))

        '''
        for i in regions[count % len(regions)] :
            draw.rectangle([(i[0], i[1]), (i[2], i[3])], outline=(0,150,0))
            count += 1
        '''
        cv2.imshow(windowName, np.array(thumb))

        ch = chr(cv2.waitKey(1) & 255)
        
        # 'q' drops out of the loop
        if ch== 'q' :
            # cleanup
            cv2.destroyWindow(windowName)
            break
        elif ch == 'd' :
            matchSelect = True
        elif ch == 'f' :
            matchSelect = False
