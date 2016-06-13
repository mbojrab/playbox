from six.moves import cPickle
import gzip
def writePickleZip (outputFile, data, log=None) :
    '''Utility to write a pickle to disk.
       outputFile : Name of the file to write. The extension should be pkl.gz
       data       : Data to write to the file
       log        : Logger to use
    '''
    if not outputFile.endswith('pkl.gz') :
        raise Exception('The file must end in the pkl.gz extension.')
    if log is not None :
        log.info('Compressing to [' + outputFile + ']')
    with gzip.open(outputFile, 'wb') as f :
        f.write(cPickle.dumps(data, protocol=cPickle.HIGHEST_PROTOCOL))

def readPickleZip (inFile, log=None) :
    '''Utility to read a pickle in from disk.
       inFile : Name of the file to read. The extension should be pkl.gz
       log    : Logger to use
    '''
    if not inFile.endswith('pkl.gz') :
        raise Exception('The file must end in the pkl.gz extension.')
    if log is not None :
        log.info('Load the data into memory')
    with gzip.open(inFile, 'rb') as f :
        data = cPickle.load(f)
    return data
