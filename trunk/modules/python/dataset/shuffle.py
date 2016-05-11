def naiveShuffle(x, log=None) :
    '''Randomize the dataset, which enforces stochasticity in training'''
    import random
    if log is not None :
        log.info('Shuffling the Elements')
    random.shuffle(x)
