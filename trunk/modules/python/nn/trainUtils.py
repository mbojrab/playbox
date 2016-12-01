import os
import numpy as np
from time import time
from dataset.writer import buildPickleInterim, buildPickleFinal, \
                           resumeEpoch, resumeLayer
from nn.net import TrainerNetwork

def renameBestNetwork(lastSave, bestNetwork, log=None) :
    if log is not None :
        log.info('Renaming Best Network to [' + bestNetwork + ']')

    if os.path.exists(bestNetwork) :
        os.remove(bestNetwork)
    os.rename(lastSave, bestNetwork)

    return bestNetwork

def _train(network, appName, dataPath, numEpochs=5, stop=1,
           synapse=None, base=None, dropout=None,
           learnC=None, learnF=None, contrF=None, momentum=None,
           kernel=None, neuron=None, numLayers=1, maxEpoch=np.inf,
           log=None) :
    '''This trains a stacked autoencoder in a greedy layer-wise manner. This
       starts by train each layer in sequence for the specified number of
       epochs, then returns the network. This can be used to initialize a
       Neural Network into a decent initial state.

       network : StackedAENetwork to used for training
       return  : Path to the trained network. This will be used as a
                 pre-trainer for the Neural Network
    '''
    isSup = lambda x: isinstance(x, TrainerNetwork)
    isImproved = lambda x, y, net : x > y if isSup(net) else x < y
    getLayer = lambda x, net: None if isSup(net) else x

    globalEpoch = lastBest = resumeEpoch(synapse)
    resetLayer = resumeLayer(synapse)
    lastSave = buildPickleInterim(
        base=base, epoch=lastBest, dropout=dropout,
        learnC=learnC, learnF=learnF, contrF=contrF,
        momentum=momentum, kernel=kernel, neuron=neuron,
        layer=getLayer(resetLayer, network))
    if synapse is None :
        network.save(lastSave)

    # this loop allows the network to be trained layer-wise. The unsupervised
    # learning method allows each layer to be trained individually.
    for layerIndex in range(numLayers) :

        # this advances training to where it left off
        if not isSup(network) and layerIndex < resetLayer :
            continue

        degraded = 0
        running = network.checkAccuracy() if isSup(network) else \
                  network.checkReconstructionLoss(layerIndex)

        # continue training until the network hits the early stoppage condition
        localEpoch = 0
        while localEpoch < maxEpoch :

            # train the network and log the result
            timer = time()

            if isSup(network) :
                globalEpoch = network.trainEpoch(globalEpoch, numEpochs)
                current = network.checkAccuracy()
                log.info('Checking Accuracy - {0}s ' \
                         '\n\tCorrect   : {1}% \n\tIncorrect : {2}%'.format(
                         time() - timer, current, (100-current)))
            else :
                globalEpoch, _ = network.trainEpoch(layerIndex, globalEpoch,
                                                    numEpochs)
                current = network.checkReconstructionLoss(layerIndex)
                log.info('Checking Loss - {0}s ' \
                         '\n\tCost   : {1}'.format(time() - timer, current))

            # check if we've improved
            if isImproved(current, running, network) :

                # reset and save the network
                degraded = 0
                running = current
                lastBest = globalEpoch
                lastSave = buildPickleInterim(
                    base=base, epoch=lastBest, dropout=dropout,
                    learnC=learnC, learnF=learnF, contrF=contrF,
                    momentum=momentum, kernel=kernel, neuron=neuron,
                    layer=getLayer(layerIndex, network))
                network.save(lastSave)
            else :
                # increment the number of lesser performing runs
                degraded += 1

            localEpoch += numEpochs

            # stopping conditions for early stoppage
            stopCondition = 100. if isSup(network) else 0.
            if degraded > int(stop) or running == stopCondition :
                break

    # rename the network which achieved the highest accuracy
    bestNetwork = buildPickleFinal(
        base=base, appName=appName, dataName=os.path.basename(dataPath),
        epoch=lastBest, accuracy=running if isSup(network) else None)
    return renameBestNetwork(lastSave, bestNetwork, log)


def trainUnsupervised(network, appName, dataPath, numEpochs=5, stop=1,
                      synapse=None, base=None, dropout=None,
                      learnC=None, learnF=None, contrF=None, momentum=None,
                      kernel=None, neuron=None, maxEpoch=np.inf,
                      log=None) :
    '''This trains a stacked autoencoder in a greedy layer-wise manner. This
       starts by train each layer in sequence for the specified number of
       epochs, then returns the network. This can be used to initialize a
       Neural Network into a decent initial state.

       network : StackedAENetwork to used for training
       return  : Path to the trained network. This will be used as a
                 pre-trainer for the Neural Network
    '''
    _train(network=network, appName=appName, dataPath=dataPath,
           numEpochs=numEpochs, stop=stop, synapse=synapse, base=base,
           dropout=dropout, learnC=learnC, learnF=learnF, contrF=contrF,
           momentum=momentum, kernel=kernel, neuron=neuron,
           numLayers=network.getNumLayers()+1, maxEpoch=maxEpoch, log=log)

def trainSupervised (network, appName, dataPath, numEpochs=5, stop=30,
                     synapse=None, base=None, dropout=None,
                     learnC=None, learnF=None, momentum=None,
                     kernel=None, neuron=None, maxEpoch=np.inf,
                     log=None) :
    '''This trains a Neural Network with early stoppage.

       network : StackedAENetwork to used for training
       return  : Path to the trained network. This will be used as a
                 pre-trainer for the Neural Network
    '''
    _train(network=network, appName=appName, dataPath=dataPath,
           numEpochs=numEpochs, stop=stop, synapse=synapse, base=base,
           dropout=dropout, learnC=learnC, learnF=learnF,
           momentum=momentum, kernel=kernel, neuron=neuron,
           maxEpoch=maxEpoch, log=log)
