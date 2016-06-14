
def buildPickleInterim(base, epoch, dropout=None, learnC=None, learnF=None, 
                       contrF=None, momentum=None, kernel=None, 
                       neuron=None, layer=None) :
    '''Create a structured name for the intermediate synapse.'''
    outName = base
    if dropout is not None :
        outName += '_dropout'+ str(dropout)
    if learnC is not None :
        outName += '_learnC'+ str(learnC)
    if learnF is not None :
        outName += '_learnF'+ str(learnF)
    if contrF is not None :
        outName += '_contrF'+ str(contrF)
    if momentum is not None :
        outName += '_momentum'+ str(momentum)
    if kernel is not None :
        outName += '_kernel'+ str(kernel)
    if neuron is not None :
        outName += '_neuron'+ str(neuron)
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
