from distill.net import DistilleryClassifier
from dataset.ingest.distill import distillKnowledge

'''This application will distill dark knowledge out of existing networks and
   into a pickled dataset which can be used as training for smaller deployable
   networks. This step should be used once a deep network has been trained to
   identify objects. Since deep networks are cumbersome and expensive, this
   technique works to make a lighter-weight deployable network. 
'''
if __name__ == '__main__' :
    import argparse
    from nn.profiler import setupLogging, Profiler

    parser = argparse.ArgumentParser()
    parser.add_argument('--log', dest='logfile', type=str, default=None,
                        help='Specify log output file.')
    parser.add_argument('--level', dest='level', default='INFO', type=str, 
                        help='Log Level.')
    parser.add_argument('--prof', dest='profile', type=str, 
                        default='Application-Profiler.xml',
                        help='Specify profile output file.')
    parser.add_argument('--softness', dest='softness', type=float, default=3,
                        help='Softness factor in softmax function.')
    parser.add_argument('--holdout', dest='holdout', type=float, default=.05,
                        help='Percent of data to be held out for testing.')
    parser.add_argument('--batch', dest='batchSize', type=int, default=50,
                        help='Batch size for training and test sets.')
    parser.add_argument('--base', dest='base', type=str, default='./distillery',
                        help='Base name of the network output and temp files.')
    parser.add_argument('--syn', dest='syn', type=str, required=True,
                        help='Synapse for the deep network to distill. This ' +
                        'network should be trained and ready.')
    parser.add_argument('data', help='Directory or pkl.gz file for the ' +
                                     'training and test sets')
    options = parser.parse_args()

    # setup the logger
    logName = 'distillToDarkPickle: ' + options.data
    log = setupLogging(logName, options.level, options.logfile)
    prof = Profiler(log=log, name=logName, profFile=options.profile)

    # distill knowledge out of the deep network into a pickle
    deepNet = DistilleryClassifier(filepath=options.syn,
                                   softmaxTemp=options.softness,
                                   prof=prof)
    distillKnowledge(deepNet=deepNet, filepath=options.data,
                     batchSize=options.batchSize, 
                     holdoutPercentage=options.holdout,  log=log)
