import argparse
from dataset.ingest.labeled import ingestImagery
from builder.args import addLoggingParams, addUnsupDataParams, \
                         addDebuggingParams
from builder.profiler import setupLogging

def buildCSV(csvFile, nets, target, data, percReturned=100., debug=False) :
    '''Test the imagery for how close it is to the target data. This also sorts
       the results according to closeness, so we can create a tiled tip-sheet.
    '''
    import numpy as np
    import pandas as pd

    # reload the target directory into the networks
    [net.loadFeatureMatrix(target) for net in nets]

    # split the data
    imagery = data[0]
    labels = data[1]

    # setup the dataframe with appropriate column names
    cols = ['Similarity', 'Batch', 'Index', 'True Label']
    cols.extend(['F' + str(ii) for ii in \
                 xrange(np.prod(nets[0].getNetworkOutputSize()[1:]))])
    sims = pd.DataFrame()

    batchSize = imagery.shape[1]
    for ii, batch in enumerate(imagery) :
        # only store the encoding from the first network
        cos, enc = nets[0].closenessAndEncoding(batch)

        # average the results found by multiple networks
        [net.closeness(batch, cos) for net in nets[1:]]
        cos /= float(len(nets))

        # collect the row for each entry of the batch
        batchList = []
        for jj in xrange(batchSize) :
            tmpList = [cos[jj], ii, jj, labels[ii][jj]]
            tmpList.extend(enc[jj])
            batchList.append(tmpList)

        # write a batch of data to the data frame
        sims = sims.append(batchList)

    # write the output to a csv
    sims.to_csv(csvFile)#, columns=cols)

if __name__ == '__main__' :
    '''This application tests how close the examples are to a provided target
       set. This averages the results against several networks and then
       reorders the chips according to the most likeness to the target set.
    '''

    # add the parent directory to grab the common functions
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from classifySAEApp import createNetworks

    parser = argparse.ArgumentParser()
    addLoggingParams(parser)
    addUnsupDataParams(parser, 'saeClass', multiLoad=True)
    addDebuggingParams(parser)
    parser.add_argument('--percent', dest='percentReturned', type=float,
                        default=100.,
                        help='Return some percentage of the highest related ' +
                             'examples. All others will not be returned to ' +
                             'the user.')
    parser.add_argument('--csv', dest='csvFile',type=str, default='output.csv',
                        help='Name of the CSV file output.')
    options = parser.parse_args()

    # setup the logger
    log, prof = setupLogging(options, 'SAE-Classification Benchmark')

    # NOTE: The pickleDataset will silently use previously created pickles if
    #       one exists (for efficiency). So watch out for stale pickles!
    train, test, labels = ingestImagery(filepath=options.data, shared=False,
                                        batchSize=options.batchSize,
                                        holdoutPercentage=options.holdout,
                                        log=log)

    # load all networks initialized to the target imagery
    nets = createNetworks(options.maxTarget, options.batchSize,
                          options.synapse, prof, options.debug)

    # test the training data for similarity to the target
    sortDataset(options.csvFile. nets, options.targetDir, test,
                options.percentReturned, options.debug)
