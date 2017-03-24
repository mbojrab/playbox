from six.moves import cPickle
import gzip
def writePickleZip (outputFile, data, log=None) :
    '''Utility to write a pickle to disk.

       NOTE: This first attempts to use protocol 2 for greater portability 
             across python versions. If that fails it, and we're using py3,
             we attempt again with the highest protocol available. Advances in
             py3.x allow large datasets to be pickled.

       outputFile : Name of the file to write. The extension should be pkl.gz
       data       : Data to write to the file
       log        : Logger to use
    '''
    if not outputFile.endswith('pkl.gz') :
        raise Exception('The file must end in the pkl.gz extension.')
    if log is not None :
        log.info('Compressing to [' + outputFile + ']')

    # attempt to pickle with protocol 2 --
    # protocol 2 is the highest protocol supported in py2.x. If we can
    # get away with pickling with this protocol, it will provide better
    # portability across python releases.
    try :
        with gzip.open(outputFile, 'wb') as f :
            f.write(cPickle.dumps(data, protocol=2))

    # TODO: find exact error thrown while pickling large networks
    except Exception as ex :
        import sys
        # large objects cannot be pickled in py2.x, so if we're using py3.x,
        # let's attempt the pickle again with the current highest.
        if sys.version_info >= (3, 0) :
            with gzip.open(outputFile.replace('pkl', 'pkl3'), 'wb') as f :
                f.write(cPickle.dumps(data, protocol=cPickle.HIGHEST_PROTOCOL))
        else : raise ex

def readPickleZip (inFile, log=None) :
    '''Utility to read a pickle in from disk.
       inFile : Name of the file to read. The extension should be pkl.gz
       log    : Logger to use
    '''
    if not inFile.endswith('pkl.gz') and not inFile.endswith('pkl3.gz') :
        raise Exception('The file must end in the pkl*.gz extension.')
    if log is not None :
        log.info('Load the data into memory')
    with gzip.open(inFile, 'rb') as f :
        # NOTE: There is a compatibility issue with pickle and numpy. py2
        #       encodes as raw bytes and py3 uses unicode. For raw strings
        #       using the proper 'encoding' option returns the string properly.
        #       py2 however makes no differentiation when encoding
        #       numpy.arrays, and these are also encoded as raw bytes. Using
        #       the 'encoding' with these objects mangles them, and there no
        #       ability to pick and choose when to utilize the encoding. 
        #       For now transfering pickled networks between py2 and py3 is
        #       not supported.
        # TODO: There are two solutions, decode as 'bytes' in py3, and perform
        #       the dencoding after the fact on all required objects, or use
        #       another archival format other than pickle.
        data = cPickle.load(f)
    return data
