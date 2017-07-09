
def stringize(x) :
    return str(x) if type(x) is not list and type(x) is not tuple else \
           '-'.join(x)

def buildPickleInterim(base, epoch, layer=None) :
    '''Create a structured name for the intermediate synapse.'''
    outName = base
    if layer is not None :
        outName += '_layer'+ str(layer)
    outName += '_epoch' + str(epoch) + '.pkl.gz'
    return outName

def buildPickleFinal(base, appName, dataName, epoch, 
                     accuracy=None, cost=None) :
    '''Create a structured name for the final synapse.'''
    from os.path import splitext, basename
    outName = base + 'Final_' + \
              str(splitext(basename(appName))[0]) + '_' + str(dataName)
    if accuracy is not None :
        outName += '_acc'+ str(accuracy)
    if cost is not None :
        outName += '_cost'+ str(cost)
    outName += '_epoch' + str(epoch) + '.pkl.gz'
    return outName

def resumeEpoch(synapse) :
    '''Return the epoch number for this synapse file. If it cannot be 
       determined, zero will be returned. 
    '''
    import re
    return 0 if synapse is None or 'epoch' not in synapse else \
           int(re.findall(r'(?<=epoch)\d+', synapse)[0])

def resumeLayer(synapse) :
    '''Return the current layer for this synapse file. If it cannot be 
       determined, zero will be returned. 
    '''
    import re
    return 0 if synapse is None or 'layer' not in synapse else \
           int(re.findall(r'(?<=layer)\d+', synapse)[0])
