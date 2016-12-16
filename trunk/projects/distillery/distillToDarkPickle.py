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
    from builder.args import addLoggingParams, addSupDataParams, setupLogging

    parser = argparse.ArgumentParser()
    addLoggingParams(parser)
    parser.add_argument('--softness', dest='softness', type=float, default=3,
                        help='Softness factor in softmax function.')
    addSupDataParams(parser, 'distillery')
    options = parser.parse_args()

    # setup the logger
    log, prof = setupLogging(options, 'distillToDarkPickle')

    # distill knowledge out of the deep network into a pickle
    deepNet = DistilleryClassifier(filepath=options.syn,
                                   softmaxTemp=options.softness,
                                   prof=prof)
    distillKnowledge(deepNet=deepNet, filepath=options.data,
                     batchSize=options.batchSize, 
                     holdoutPercentage=options.holdout,  log=log)
