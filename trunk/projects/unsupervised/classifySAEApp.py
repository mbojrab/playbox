import argparse
from dataset.ingest.labeled import ingestImagery
from nn.profiler import setupLogging, Profiler

def createNetworks(target, netFiles, prof) :
    '''Read and create each network initialized with the target dataset.'''
    from ae.net import ClassifierSAENetwork
    return [ClassifierSAENetwork(target, syn, prof) for syn in netFiles]

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
        cos = np.zeros((batchSize,), dtype=np.float32)
        # average the results found by multiple networks
        [net.closeness(batch, cos) for net in nets]
        sims.append(cos / float(len(nets)))

    # rank their closeness from high to low
    sims = [(ii // batchSize, ii % batchSize, sim) \
            for ii, sim in enumerate(np.concatenate(sims))]
    sims = sorted(sims, key=lambda x: x[-1], reverse=True)

    # reorder the imagery to match the ranking
    counter = 0
    numImages = int((1.-percentile) * np.prod(imagery.shape[:2]))
    sortedImagery = np.zeros([numImages] + list(imagery.shape[-3:]),
                             dtype=imagery.dtype)
    for ii, jj, sim in sims[:numImages] :
        sortedImagery[counter][:] = imagery[ii][jj][:]
        counter += 1

    # dump the ranked result as a series of batches
    if debug :
        newNumBatch = math.ceil(numImages / batchSize)
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

    return sortedImagery

if __name__ == '__main__' :
    '''This application tests how close the examples are to a provided target
       set. This averages the results against several networks and then
       reorders the chips according to the most likeness to the target set.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', dest='logfile', type=str, default=None,
                        help='Specify log output file.')
    parser.add_argument('--level', dest='level', default='INFO', type=str, 
                        help='Log Level.')
    parser.add_argument('--prof', dest='profile', type=str, 
                        default='Application-Profiler.xml',
                        help='Specify profile output file.')
    parser.add_argument('--batch', dest='batchSize', type=int, default=100,
                        help='Batch size for training and test sets.')
    parser.add_argument('--base', dest='base', type=str, default='./saeClass',
                        help='Base name of the network output and temp files.')
    parser.add_argument('--target', dest='targetDir', type=str, required=True,
                        help='Directory with target data to match.')
    parser.add_argument('--percentile', dest='percentile', type=float,
                        default=.95, help='Keep the top percentile of ' +
                        'information corresponding to the most likely matches')
    parser.add_argument('--syn', dest='synapse', type=str, nargs='+',
                        help='Load from a previously saved network.')
    parser.add_argument('--debug', dest='debug', type=bool, required=False,
                        help='Drop debugging information about the runs.')
    parser.add_argument('data', help='Directory of input imagery.')
    options = parser.parse_args()

    # setup the logger
    logName = 'SAE-Classification Benchmark:  ' + options.data
    log = setupLogging(logName, options.level, options.logfile)
    prof = Profiler(log=log, name=logName, profFile=options.profile)

    # NOTE: The pickleDataset will silently use previously created pickles if
    #       one exists (for efficiency). So watch out for stale pickles!
    train, test, labels = ingestImagery(filepath=options.data, shared=True,
                                        batchSize=options.batchSize, log=log)

    # load all networks initialized to the target imagery
    nets = createNetworks(options.targetDir, options.synapse, prof)

    # test the training data for similarity to the target
    sortDataset(nets, test[0].get_value(borrow=True), 
                options.percentile, options.debug)
