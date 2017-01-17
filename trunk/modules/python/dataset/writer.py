
def stringize(x) :
    return str(x) if type(x) is not list and type(x) is not tuple else \
           str(x).strip('[]').replace(', ', '-')

def buildPickleInterim(base, epoch, learnC=None, learnF=None, 
                       kernel=None, neuron=None, layer=None) :
    '''Create a structured name for the intermediate synapse.'''
    outName = base
    if kernel is not None :
        outName += '_kernel'+ stringize(kernel)
    if neuron is not None :
        outName += '_neuron'+ stringize(neuron)
    if learnC is not None :
        outName += '_learnC'+ stringize(learnC)
    if learnF is not None :
        outName += '_learnF'+ stringize(learnF)
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

def resumeData(synapse) :
    '''Separate the synapse data to grab the network structure.'''
    import re

    kernel = '[none]' if synapse is None or 'kernel' not in synapse else \
      [int(x) for x in re.findall(r'(?<=kernel)(.*?)_', synapse)[0].split('-')]
    neuron = '[none]' if synapse is None or 'neuron' not in synapse else \
      [int(x) for x in re.findall(r'(?<=neuron)(.*?)_', synapse)[0].split('-')]
    learnC = '[none]' if synapse is None or 'learnC' not in synapse else \
      [float(x) for x in re.findall(r'(?<=learnC)(.*?)_', synapse)[0].split('-')]
    learnF = '[none]' if synapse is None or 'learnF' not in synapse else \
      [float(x) for x in re.findall(r'(?<=learnF)(.*?)_', synapse)[0].split('-')]

    return kernel, neuron, learnC, learnF
