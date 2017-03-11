import argparse
from dataset.ingest.unlabeled import ingestImagery
from builder.args import addLoggingParams, addUnsupDataParams
from builder.profiler import setupLogging

def createNetworks(target, maxTarget, batchSize, netFiles, prof, debug) :
    '''Read and create each network initialized with the target dataset.'''
    from ae.net import ClassifierSAENetwork
    nets = [ClassifierSAENetwork(maxTarget, syn, prof, debug) \
            for syn in netFiles]
    [net.loadFeatureMatrix(target) for net in nets]
    return nets

def sortDataset(netList, imagery, percentile=.95, debug=False) :
    '''Test the imagery for how close it is to the target data. This also sorts
       the results according to closeness, so we can create a tiled tip-sheet.
    '''
    from dataset.debugger import saveTiledImage
    import numpy as np
    import math
    sims = []
    batchSize = imagery.shape[1]
    for ii, batch in enumerate(imagery) :
        # average the results found by multiple networks
        cos = np.zeros((batchSize,), dtype=np.float32)
        [net.closeness(batch, cos) for net in nets]
        sims.append(cos / float(len(nets)))

    # rank their closeness from high to low
    sims = [(ii // batchSize, ii % batchSize, sim) \
            for ii, sim in enumerate(np.concatenate(sims))]
    sims = sorted(sims, key=lambda x: x[-1], reverse=True)

    # reorder the imagery to match the ranking
    numImages = int((1.-percentile) * np.prod(imagery.shape[:2]))
    sortedImagery = np.zeros([numImages] + list(imagery.shape[-3:]),
                             dtype=imagery.dtype)
    sortedConfidence = np.zeros([numImages], dtype=np.float32)
    for counter, (ii, jj, sim) in enumerate(sims[:numImages]) :
        sortedImagery[counter][:] = imagery[ii][jj][:]
        sortedConfidence[counter] = sim

    # dump the ranked result as a series of batches
    if debug :
        newNumBatch = int(math.ceil(numImages / batchSize))
        newFlatShape = [newNumBatch * batchSize] + list(imagery.shape[-3:])
        newBatchShape = [newNumBatch, batchSize] + list(imagery.shape[-3:])
        dimdiff = tuple([(0, a - b) for a, b in zip(newFlatShape,
                                                    sortedImagery.shape)])
        tmp = np.pad(sortedImagery, dimdiff, 
                     mode='constant', constant_values=0)
        tmp = np.reshape(tmp, newBatchShape)
        for ii in range(len(imagery)) :
            saveTiledImage(imagery[ii], str(ii) + '.tif',
                           imagery.shape[-2:])
        for ii in range(len(tmp)) :
            saveTiledImage(tmp[ii], str(ii) + '_sorted.tif',
                           imagery.shape[-2:])

    return sortedImagery, sortedConfidence

if __name__ == '__main__' :
    '''This application tests how close the examples are to a provided target
       set. This averages the results against several networks and then
       reorders the chips according to the most likeness to the target set.
    '''
    parser = argparse.ArgumentParser()
    addLoggingParams(parser)
    addUnsupDataParams(parser, 'saeClass', multiLoad=True)
    parser.add_argument('--percentile', dest='percentile', type=float,
                        default=.95, help='Keep the top percentile of ' +
                        'information corresponding to the most likely matches')
    parser.add_argument('--debug', dest='debug', type=bool, required=False,
                        help='Drop debugging information about the runs.')
    options = parser.parse_args()

    # setup the logger
    log, prof = setupLogging(options, 'SAE-Classification Benchmark')

    # NOTE: The pickleDataset will silently use previously created pickles if
    #       one exists (for efficiency). So watch out for stale pickles!
    train, test = ingestImagery(filepath=options.data, shared=True,
                                batchSize=options.batchSize,
                                holdoutPercentage=options.holdout,
                                log=log)

    # load all networks initialized to the target imagery
    nets = createNetworks(options.targetDir, options.maxTarget,
                          options.batchSize, options.synapse,
                          prof, options.debug)

    # test the training data for similarity to the target
    sortDataset(nets, test.get_value(borrow=True), 
                options.percentile, options.debug)
