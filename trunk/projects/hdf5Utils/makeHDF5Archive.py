import os
import argparse
from nn.profiler import setupLogging
from dataset.hdf5 import archiveDirToHDF5

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', dest='logfile', type=str, default=None,
                        help='Specify log output file.')
    parser.add_argument('--level', dest='level', default='INFO', type=str, 
                        help='Log Level.')
    parser.add_argument('--h5', type=str, help='Override the HDF5 name.')
    parser.add_argument('data', type=str,
                        help='Directory to store in the archive')
    options = parser.parse_args()

    # setup the logger
    logName = 'Archive to HDF5:  ' + options.data
    log = setupLogging(logName, options.level, options.logfile)

    # name the file the same as the directory by default
    if options.h5 is None :
        splitPath = os.path.split(options.data)
        options.h5 = os.path.join(splitPath[0], splitPath[1] + '.hdf5')

    # write the HDF5
    archiveDirToHDF5(options.h5, options.data, log=log)
