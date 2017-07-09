from ae.net import TrainerSAENetwork, ClassifierSAENetwork
from builder.profiler import setupLogging
from builder.sae import setupCommandLine, buildNetwork
from dataset.ingest.unlabeled import ingestImagery
from nn.trainUtils import trainUnsupervised

from dataset.shared import getShape

tmpNet = './local.pkl.gz'

if __name__ == '__main__' :
    '''Build and train an SAE, then test against a target example.'''
    import argparse
    from builder.args import addLoggingParams, \
                             addDebuggingParams, \
                             addEarlyStop, \
                             addUnsupDataParams, \
                             addUnsupContiguousParams
    import sys, os
    import numpy as np
    from builder.sae import addContiguousAE
    from nn.contiguousLayer import ContiguousLayer

    # add the parent directory to grab the common functions
    sys.path.append(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from classifySAETrainer import testCloseness

    # setup the common command line options
    parser = argparse.ArgumentParser()
    addLoggingParams(parser)
    addDebuggingParams(parser)
    addEarlyStop(parser)
    addUnsupDataParams(parser, 'retrainContig', multiLoad=False)
    addUnsupContiguousParams(parser)
    parser.add_argument('--layer', dest='layer', type=int, default=None,
                        help='Specify the layer to start training. A value' +
                             ' of None uses the layer in the synapse file.')
    parser.add_argument('--remove', dest='remove', type=int, default=None,
                        help='Specify the number of layers to remove.')
    options = parser.parse_args()

    if options.synapse is None :
        raise ValueError('Please specify a Synapse file to start.')

    # setup the logger
    log, prof = setupLogging (options, 'SAE-Classification Benchmark')

    # NOTE: The pickleDataset will silently use previously created pickles if
    #       one exists (for efficiency). So watch out for stale pickles!
    train, test = ingestImagery(filepath=options.data, shared=True,
                                batchSize=options.batchSize, 
                                holdoutPercentage=options.holdout,
                                log=log)

    # create the stacked network
    trainer = TrainerSAENetwork(train, test, options.synapse,
                                options.greedyNet, prof, options.debug)

    # remove the contiguous layers
    count = options.remove if options.remove is not None else np.inf
    while count and isinstance(trainer.getLayer(-1), ContiguousLayer) :
        trainer.removeLayer()
        count -= 1

    # rebuild the contiguous layers -- based on the new specifications
    addContiguousAE (trainer, getShape(train)[1:], options, prof=prof)

    # train the SAE
    trainUnsupervised(trainer, __file__, options.data,
                      numEpochs=options.limit, stop=options.stop,
                      synapse=options.synapse, base=options.base,
                      maxEpoch=options.epoch,
                      resetLayer=options.layer, log=log)
    trainer.save(tmpNet)
    options.synapse = tmpNet

    # train the SAE
    net = ClassifierSAENetwork(options.maxTarget, options.synapse, prof,
                               options.debug)
    net.loadFeatureMatrix(options.targetDir)

    # test the training data for similarity to the target
    testCloseness(net, test.get_value(borrow=True))
