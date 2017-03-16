from ae.net import TrainerSAENetwork, ClassifierSAENetwork
from builder.profiler import setupLogging
from builder.sae import setupCommandLine, buildNetwork
from dataset.ingest.unlabeled import ingestImagery
from nn.trainUtils import trainUnsupervised

from dataset.shared import getShape

tmpNet = './local.pkl.gz'

if __name__ == '__main__' :
    '''Build and train an SAE, then test against a target example.'''
    from ..classifySAETrainer import testCloseness
    from nn.contiguousLayer import ContiguousLayer

    options = setupCommandLine()

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
    trainer = TrainerSAENetwork(train, test, options.synapse, prof,
                                options.debug)

    # remove the contiguous layers
    while isinstance(trainer.getLayer(-1), ContiguousLayer) :
        trainer.removeLayer()

    # rebuild the contiguous layers -- based on the new specifications
    options.kernel = []
    options.kernelSize = []
    buildNetwork(trainer, getShape(train)[1:], options, prof=prof)

    # train the SAE
    trainUnsupervised(trainer, __file__, options.data,
                      numEpochs=options.limit, stop=options.stop,
                      synapse=options.synapse, base=options.base,
                      kernel=options.kernel, neuron=options.neuron,
                      learnC=options.learnC, learnF=options.learnF,
                      maxEpoch=options.epoch, log=log)
    trainer.save(tmpNet)
    options.synapse = tmpNet

    # train the SAE
    net = ClassifierSAENetwork(options.maxTarget, options.synapse, prof,
                               options.debug)
    net.loadFeatureMatrix(options.targetDir)

    # test the training data for similarity to the target
    testCloseness(net, test.get_value(borrow=True))
