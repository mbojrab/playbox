import os
import argparse
from builder.args import addLoggingParams
from builder.profiler import setupLogging
from dataset.hdf5 import archiveDirToHDF5

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    addLoggingParams(parser)
    parser.add_argument('data', type=str,
                        help='Directory to store in the archive')
    options = parser.parse_args()

    # setup the logger
    log, prof = setupLogging(options, 'Archive to HDF5')

    # name the file the same as the directory by default
    splitPath = os.path.split(options.data)
    h5File = os.path.join(splitPath[0], splitPath[1], splitPath[1] + '.hdf5')

    # write the HDF5
    archiveDirToHDF5(h5File, options.data, log=log)
