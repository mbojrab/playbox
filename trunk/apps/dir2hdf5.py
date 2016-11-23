
import argparse
import sys
import os
import os.path as osp

import h5py
import numpy as np

from nn.profiler import setupLogging
from dataset.reader import readImage

STORAGE_TYPE = np.uint8

def dir2hdf5(directory, outputfile, convert, log):
    ''' Convert a directory to an hdf5 file with the same content '''

    log.info('Processing {0} into {1}'.format(directory, outputfile))

    with h5py.File(outputfile, 'a') as h5file:

        fidx = 0
        for root, dirs, files in os.walk(directory):

            # Create groups for directories
            for d in (osp.join(root, d) for d in dirs):
                dset = h5file.create_group(d)

            for f in (osp.join(root, f) for f in files):

                # replace old contents with new
                if f in h5file:
                    del h5file[f]
                if convert:
                    blob, typeinfo = readImage(f, None, raw=True)
                else:
                    blob = np.fromfile(f, dtype=STORAGE_TYPE)
                    typeinfo = None

                dset = h5file.create_dataset(f, data=blob,
                                             compression='gzip')
                if typeinfo is not None:
                    dset.attrs['type'] = typeinfo
                  
                fidx += 1
                if fidx % 1024 == 0: 
                    log.debug('flushing file')
                    h5file.flush()

        h5file.flush()


def hdf52dir(inputfile, outputdirectory, log):
    ''' Dump an hdf5 to the filesystem '''

    log.info('Dumping {0} to {1}'.format(inputfile, outputdirectory))
    
    def createtree(name, item):
        newthing = osp.join(outputdirectory, name)
        if isinstance(item, h5py.Group):
            # Create the full directory path
            os.makedirs(newthing)
        elif isinstance(item, h5py.Dataset):
            # Load the h5data and write it to the file
            content = np.frombuffer(item[:], dtype=STORAGE_TYPE)
            content.tofile(newthing)

    with h5py.File(inputfile, 'r') as h5file:
        h5file.visititems(createtree)
        


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--log', dest='logfile', type=str, default=None,
                        help='Specify log output file.')
    parser.add_argument('--level', dest='level', default='INFO', type=str, 
                        help='Log Level.')
    parser.add_argument('--convert', dest='convert', action='store_true',
                        help='Convert image file to numpy arrays before'
                              ' storinng in the hdf5')
    parser.add_argument('--reverse', dest='reverse', action='store_true',
                        help='Run process in reverse and create directory from'
                              ' the hdf5')
    parser.add_argument('directory', help='Directory/File to convert')
    parser.add_argument('outputfile', nargs='?', type=str, default=None,
                        help='Specify the output directory.')
    options = parser.parse_args()
    
    logName = 'dir2h5: ' + options.directory
    log = setupLogging(logName, options.level, options.logfile)

    reverse = options.reverse
    if not reverse:
        # Create the hdf5
        directory = options.directory
        outputfile = options.outputfile
        # Ensure the output file is valid
        if outputfile is None:
            outputfile = osp.basename(options.directory)
        if not outputfile.endswith('.hdf5'):
            outputfile += '.hdf5'
        dir2hdf5(directory, outputfile, options.convert, log)
    else:
        # Extract the file contents to a directory tree
        inputfile = options.directory
        outputdirectory = options.outputfile
        if osp.exists(outputdirectory):
            raise Exception('{0} already exists'.format(outputdirectory))
        
        hdf52dir(inputfile, outputdirectory, log)
        
        




