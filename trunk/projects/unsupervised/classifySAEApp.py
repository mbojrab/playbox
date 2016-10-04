import argparse
from dataset.ingest.labeled import ingestImagery
from nn.profiler import setupLogging, Profiler
from dataset.minibatch import makeContiguous

def readTargetData(targetpath) :
    '''Read a directory of data to use as a feature matrix.'''
    import os
    from dataset.reader import readImage
    return makeContiguous([(readImage(os.path.join(targetpath, im))) \
                           for im in os.listdir(targetpath)])[0]

def createNetworks(target, netFiles, prof) :
    '''Read and create each network initialized with the target dataset.'''
    from ae.net import ClassifierSAENetwork
    return [ClassifierSAENetwork(target, syn, prof) for syn in netFiles]

def testCloseness(netList, imagery) :
    '''Test the imagery for how close it is to the target data. This also sorts
       the results according to closeness, so we can create a tiled tip-sheet.
    '''
    from dataset.debugger import saveTiledImage
    import numpy as np
    sims = []
    batchSize = imagery.shape[1]
    for ii, batch in enumerate(imagery) :
        # average the results found by multiple networks
        sims.append(np.mean([net.closeness(batch) for net in nets], axis=0))


    # rank their closeness from high to low
    sims = [(ii // batchSize, ii % batchSize, sim) \
            for ii, sim in enumerate(np.concatenate(sims))]
    sims = sorted(sims, key=lambda x: x[-1], reverse=True)

    # reorder the imagery to match the ranking
    sortedImagery = np.ndarray(imagery.shape, dtype=np.float32)
    counter = 0
    for ii, jj, sim in sims :
        sortedImagery[counter // batchSize][counter % batchSize][:] = \
            imagery[ii][jj][:]
        counter += 1

    # dump the ranked result as a series of batches
    for ii in range(len(imagery)) :
        saveTiledImage(imagery[ii], str(ii) + '.tif', (28,28))
        saveTiledImage(sortedImagery[ii], str(ii) + '_sorted.tif', (28,28))

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
    parser.add_argument('--syn', dest='synapse', type=str, nargs='+',
                        help='Load from a previously saved network.')
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

    # load example imagery --
    # these are confirmed objects we are attempting to identify 
    target = readTargetData(options.targetDir)

    nets = createNetworks(target, options.synapse, prof)

    # test the training data for similarity to the target
    testCloseness(nets, test[0].get_value(borrow=True))
