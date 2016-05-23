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

def trainUnsupervised(network, appName, dataPath, numEpochs=5, stop=30, 
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
    # first we pre-train the data and at each epoch, we save it to disk
    lastSave = ''
    globalEpoch = 0
    for layerIndex in range(network.getNumLayers()) :
        globalEpoch, cost = network.trainEpoch(layerIndex, globalEpoch, 
                                               numEpochs)
        lastSave = buildPickleInterim(base=base,
                                      epoch=globalEpoch,
                                      dropout=dropout,
                                      learnC=learnC,
                                      learnF=learnF,
                                      contrF=contrF,
                                      kernel=kernel,
                                      neuron=neuron,
                                      layer=layerIndex)
        network.save(lastSave)

    # rename the network which achieved the highest accuracy
    bestNetwork = buildPickleFinal(base=base, appName=appName, 
                                   dataName=os.path.basename(dataPath), 
                                   epoch=numEpochs)
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
    globalCount = lastBest = resumeEpoch(synapse)
    runningAccuracy = 0.0
    lastSave = ''
    while True :
        timer = time()

        # run the specified number of epochs
        globalCount = network.trainEpoch(globalCount, numEpochs)
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
            lastBest = globalCount
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
