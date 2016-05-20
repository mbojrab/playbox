def distillKnowledge(deepNet, filepath, batchSize=50, 
                     holdoutPercentage=0.5, log=None) :
    import os
    from dataset.ingest.labeled import ingestImagery
    from dataset.pickle import writePickleZip

    # build a new pickle with this information
    # TODO: NETWORKS NEED UNIQUE LABEL IDENTIFIERS WHICH CAN BE ADDED HERE
    rootpath = os.path.abspath(filepath)
    outputFile = os.path.join(rootpath, os.path.basename(rootpath) + 
                              '_darkLabels_' +
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
    darkLabels = [deepNet.classifyAndSoftmax(dataset) \
                  for dataset in train[0]]
    train = train[0], darkLabels

    # pickle the dataset
    writePickleZip(outputFile, (train, test, labels), log)

    # return the output filename
    return outputFile
