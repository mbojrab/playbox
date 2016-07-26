import argparse

from nn.net import ClassifierNetwork, TrainerNetwork
from dataset.pickle import readPickleZip
from dataset.shared import splitToShared
from nn.trainUtils import trainSupervised
from nn.profiler import setupLogging, Profiler

'''This application will distill dark knowledge out of existing networks and
   into a pickled dataset which can be used as training for smaller deployable
   networks. This step should be used once a deep network has been trained to
   identify objects. Since deep networks are cumbersome and expensive, this
   technique works to make a lighter-weight deployable network. 
'''
if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    parser.add_argument('--log', dest='logfile', type=str, default=None,
                        help='Specify log output file.')
    parser.add_argument('--level', dest='level', default='INFO', type=str, 
                        help='Log Level.')
    parser.add_argument('--prof', dest='profile', type=str, 
                        default='Application-Profiler.xml',
                        help='Specify profile output file.')
    parser.add_argument('--limit', dest='limit', type=int, default=5,
                        help='Number of runs between validation checks.')
    parser.add_argument('--stop', dest='stop', type=int, default=5,
                        help='Number of inferior validation checks to end.')
    parser.add_argument('--softness', dest='softness', type=float, default=3,
                        help='Softness factor in softmax function.')
    parser.add_argument('--holdout', dest='holdout', type=float, default=.05,
                        help='Percent of data to be held out for testing.')
    parser.add_argument('--batch', dest='batchSize', type=int, default=50,
                        help='Batch size for training and test sets.')
    parser.add_argument('--base', dest='base', type=str, default='./distillery',
                        help='Base name of the network output and temp files.')
    parser.add_argument('--shallow', dest='shallow', type=str, required=True,
                        help='Synapse for the shallow target network. This ' +
                        'network should be populated with freshly ' + 
                        'initialized layers for optimal results.')
    parser.add_argument('--deep', dest='deep', type=str, default=None,
                        help='Synapse for the deep network to distill. This ' +
                        'network should be trained and ready.')
    parser.add_argument('--dark', dest='dark', type=str, default=None,
                        help='pkl.gz file previously created by the ' +
                        'distillery for dark knowledge transfer.')
    parser.add_argument('--data', dest='data', type=str, default=None,
                        help='Directory or pkl.gz file for the training and ' +
                        'test sets')
    options = parser.parse_args()

    # setup the logger
    logName = 'distillery: ' + options.data
    log = setupLogging(logName, options.level, options.logfile)
    prof = Profiler(log=log, name=logName, profFile=options.profile)

    # if the user specified a deep network and dataset, then distill the
    # knowledge into a new pickle to use for training.
    if options.deep is not None :
        from dataset.ingest.distill import distillKnowledge

        if options.dark is not None :
            raise Exception('Only specify one usage, --deep or --dark.')
        if options.data is None :
            raise Exception('When specifying a deep network, please also ' +
                            'specify a dataset to distill using --data.')

        # distill knowledge out of the deep network into a pickle
        deepNet = ClassifierNetwork(filepath=options.deep,
                                    softmaxTemp=options.softness,
                                    prof=prof)
        options.dark = distillKnowledge(deepNet=deepNet,
                                        filepath=options.data,
                                        batchSize=options.batchSize, 
                                        holdoutPercentage=options.holdout, 
                                        log=log)

    # use the pickle to train a shallower network to perform the same task
    train, test, labels = readPickleZip(options.dark, log)
    shallowNet = TrainerNetwork(splitToShared(train, castLabelInt=False), 
                                splitToShared(test), labels,
                                filepath=options.shallow,
                                softmaxTemp=options.softness, prof=prof)

    trainSupervised(shallowNet, __file__, options.data, 
                    numEpochs=options.limit, stop=options.stop, 
                    synapse=options.shallow, base=options.base, log=log)
