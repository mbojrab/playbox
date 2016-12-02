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
    parser.add_argument('data', type=str,
                        help='Directory to store in the archive')
    options = parser.parse_args()

    # setup the logger
    logName = 'Archive to HDF5:  ' + options.data
    log = setupLogging(logName, options.level, options.logfile)

    # name the file the same as the directory by default
    splitPath = os.path.split(options.data)
    h5File = os.path.join(splitPath[0], splitPath[1], splitPath[1] + '.hdf5')

    # write the HDF5
    archiveDirToHDF5(h5File, options.data, log=log)
