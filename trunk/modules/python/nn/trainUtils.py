import os
from dataset.writer import buildPickleInterim, buildPickleFinal, resumeEpoch
from time import time

def renameBestNetwork(lastSave, bestNetwork, log=None) :
    if log is not None :
        log.info('Renaming Best Network to [' + bestNetwork + ']')

    if os.path.exists(bestNetwork) :
        os.remove(bestNetwork)
    os.rename(lastSave, bestNetwork)

    return bestNetwork

# TODO: This is largely redundant with the method below. Look into combining
#       them in an intuitive manner. There are minor differences between them
#       so watch out for this!
def trainUnsupervised(network, appName, dataPath, numEpochs=5, stop=1,
                      synapse=None, base=None, dropout=None, 
                      learnC=None, learnF=None, contrF=None, momentum=None, 
                      kernel=None, neuron=None, log=None) :
    '''This trains a stacked autoencoder in a greedy layer-wise manner. This
       starts by train each layer in sequence for the specified number of
       epochs, then returns the network. This can be used to initialize a
       Neural Network into a decent initial state.

       network : StackedAENetwork to used for training
       return  : Path to the trained network. This will be used as a 
                 pre-trainer for the Neural Network
    '''
    # train each layer in sequence --
    # first we pre-train the data and at each epoch, we save each to disk
    #
    # the last iteration trains the network as a whole --
    # this ensures the network fine-tunes its encodings wrt to all layers'
    # encoding and decoding loss.
    globalEpoch = lastBest = resumeEpoch(synapse)
    lastSave = buildPickleInterim(base=base,
                                  epoch=lastBest,
                                  dropout=dropout,
                                  learnC=learnC,
                                  learnF=learnF,
                                  contrF=contrF,
                                  kernel=kernel,
                                  neuron=neuron,
                                  layer=0)
    network.save(lastSave)

    # train each layer individually
    # this fully trains each layer before moving to next
    for layerIndex in range(network.getNumLayers()) :
        degradationCount = 0
        runningCost = network.checkReconstructionLoss(layerIndex)

        # continue training the layer until the network the network stops
        # making progress or it hit a max limit
        while True :

            # train each layer -- greedily
            timer = time()
            globalEpoch, _ = network.trainEpoch(layerIndex, globalEpoch,
                                                numEpochs)
            elapsed = time() - timer
            curCost = network.checkReconstructionLoss(layerIndex)
            log.info('Checking Loss - {0}s - {1}'.format(elapsed, curCost))

            # check if we've improved
            if curCost < runningCost :

                # reset and save the network
                degradationCount = 0
                runningCost = curCost
                lastBest = globalEpoch
                lastSave = buildPickleInterim(base=base,
                                              epoch=lastBest,
                                              dropout=dropout,
                                              learnC=learnC,
                                              learnF=learnF,
                                              contrF=contrF,
                                              kernel=kernel,
                                              neuron=neuron,
                                              layer=layerIndex)
                network.save(lastSave)
            else :
                # increment the number of poor performing runs
                degradationCount += 1

            # stopping conditions for regularization
            if degradationCount > int(stop) or runningCost == 0. :
                break

    # rename the network which achieved the highest accuracy
    bestNetwork = buildPickleFinal(base=base, appName=appName,
                                   dataName=os.path.basename(dataPath),
                                   epoch=lastBest)
    return renameBestNetwork(lastSave, bestNetwork, log)

def trainSupervised (network, appName, dataPath, numEpochs=5, stop=30, 
                     synapse=None, base=None, dropout=None, 
                     learnC=None, learnF=None, momentum=None, 
                     kernel=None, neuron=None, log=None) :
    '''This trains a Neural Network with early stoppage.
       
       network : StackedAENetwork to used for training
       return  : Path to the trained network. This will be used as a 
                 pre-trainer for the Neural Network
    '''
    degradationCount = 0
    globalEpoch = lastBest = resumeEpoch(synapse)
    runningAccuracy = network.checkAccuracy()
    lastSave = buildPickleInterim(base=base,
                                  epoch=lastBest,
                                  dropout=dropout,
                                  learnC=learnC,
                                  learnF=learnF,
                                  momentum=momentum,
                                  kernel=kernel,
                                  neuron=neuron)
    network.save(lastSave)
    while True :
        timer = time()

        # run the specified number of epochs
        globalEpoch = network.trainEpoch(globalEpoch, numEpochs)
        # calculate the accuracy against the test set
        curAcc = network.checkAccuracy()
        log.info('Checking Accuracy - {0}s ' \
                 '\n\tCorrect   : {1}% \n\tIncorrect : {2}%'.format(
                 time() - timer, curAcc, (100-curAcc)))

        # check if we've done better
        if curAcc > runningAccuracy :
            # reset and save the network
            degradationCount = 0
            runningAccuracy = curAcc
            lastBest = globalEpoch
            lastSave = buildPickleInterim(base=base,
                                          epoch=lastBest,
                                          dropout=dropout,
                                          learnC=learnC,
                                          learnF=learnF,
                                          momentum=momentum,
                                          kernel=kernel,
                                          neuron=neuron)
            network.save(lastSave)
        else :
            # increment the number of poor performing runs
            degradationCount += 1

        # stopping conditions for regularization
        if degradationCount > int(stop) or runningAccuracy == 100. :
            break

    # rename the network which achieved the highest accuracy
    bestNetwork = buildPickleFinal(base=base, appName=appName,
                                   dataName=os.path.basename(dataPath),
                                   epoch=lastBest, accuracy=runningAccuracy)
    return renameBestNetwork(lastSave, bestNetwork, log)
