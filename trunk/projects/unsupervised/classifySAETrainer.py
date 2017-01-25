from ae.net import TrainerSAENetwork, ClassifierSAENetwork
from builder.profiler import setupLogging
from builder.sae import setupCommandLine, buildNetwork
from dataset.ingest.labeled import ingestImagery
from nn.trainUtils import trainUnsupervised
from dataset.shared import getShape

tmpNet = './local.pkl.gz'

def testCloseness(net, imagery) :
    '''Test the imagery for how close it is to the target data. This also sorts
       the results according to closeness, so we can create a tiled tip-sheet.
    '''
    from dataset.debugger import saveTiledImage
    import numpy as np
    for ii, batch in enumerate(imagery) :
        sims = net.closeness(batch)
        sims = [(jj, sim) for jj, sim in enumerate(sims)]
        sims = sorted(sims, key=lambda x: x[1], reverse=True)

        sortedBatch = np.ndarray(batch.shape, dtype=np.float32)
        for jj, sim in enumerate(sims) :
            sortedBatch[jj][:] = batch[sim[0]][:]

        imgDims = imagery.shape[-2:]
        saveTiledImage(batch, str(ii) + '.tif', imgDims)
        saveTiledImage(sortedBatch, str(ii) + '_sorted.tif', imgDims)

if __name__ == '__main__' :
    '''Build and train an SAE, then test a '''
    import numpy as np
    options = setupCommandLine()

    # setup the logger
    log, prof = setupLogging (options, 'SAE-Classification Benchmark')

    # NOTE: The pickleDataset will silently use previously created pickles if
    #       one exists (for efficiency). So watch out for stale pickles!
    train, test, labels = ingestImagery(filepath=options.data, shared=True,
                                        batchSize=options.batchSize, 
                                        holdoutPercentage=options.holdout,
                                        log=log)

    # create the stacked network
    print(options.debug)
    trainer = TrainerSAENetwork(train, test, options.synapse, prof,
                                options.debug)
    if options.synapse is None :
        buildNetwork(trainer, getShape(train[0])[1:], options, prof=prof)

    # train the SAE
    trainUnsupervised(trainer, __file__, options.data, 
                      numEpochs=options.limit, stop=options.stop,
                      synapse=options.synapse, base=options.base,
                      dropout=(options.dropoutC is not None and \
                               len(options.dropoutC) > 0),
                      learnC=options.learnC, learnF=options.learnF,
                      contrF=None, kernel=options.kernel,
                      neuron=options.neuron, maxEpoch=options.epoch, log=log)
    trainer.save(tmpNet)
    options.synapse = tmpNet

    # train the SAE
    net = ClassifierSAENetwork(options.targetDir, options.synapse, prof,
                               options.debug)

    # test the training data for similarity to the target
    testCloseness(net, test[0].get_value(borrow=True))

