def distillKnowledge(deepNet, filepath, batchSize=50, 
                     holdoutPercentage=0.5, log=None) :
    import os
    import numpy as np
    from dataset.ingest.labeled import ingestImagery
    from dataset.pickle import writePickleZip
    from distill.net import DistilleryClassifier

    if not isinstance(deepNet, DistilleryClassifier) :
        raise ValueError('The network must be setup as a DistilleryNetwork.')

    # build a new pickle with this information
    # TODO: NETWORKS NEED UNIQUE LABEL IDENTIFIERS WHICH CAN BE ADDED HERE
    rootpath = os.path.abspath(filepath)
    outputFile = os.path.join(rootpath, os.path.basename(rootpath) + 
                              '_darkLabels' +
                              '_holdout_' + str(holdoutPercentage) +
                              '_batch_' + str(batchSize) +
                              '.pkl.gz')
    if os.path.exists(outputFile) :
        if log is not None :
            log.info('Pickle exists for this dataset [' + outputFile +
                     ']. Using this instead.')
        return outputFile

    # NOTE: The pickleDataset will silently use previously created pickles if
    #       one exists (for efficiency). So watch out for stale pickles!
    train, test, labels = ingestImagery(filepath=filepath, shared=False,
                                        batchSize=batchSize, 
                                        holdoutPercentage=holdoutPercentage, 
                                        log=log)

    if log is not None :
        log.info('Distilling knowledge from deep network')

    # distill knowledge into a pickle which can be used to train other networks
    batchSize = train[0].shape[0]
    darkLabels = [deepNet.softTarget(dataset) for dataset in train[0]]
    labelDims = [len(darkLabels)] + list(darkLabels[0].shape)
    darkLabels = np.reshape(np.concatenate(darkLabels), labelDims)
    train = train[0], train[1], darkLabels

    # pickle the dataset
    writePickleZip(outputFile, (train, test, labels), log)

    # return the output filename
    return outputFile
