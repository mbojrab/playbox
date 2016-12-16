import theano.tensor as t
from numpy.random import RandomState

def addLoggingParams (parser) :
    '''Setup common logging and profiler options.'''
    parser.add_argument('--log', dest='logfile', type=str, default=None,
                        help='Specify log output file.')
    parser.add_argument('--level', dest='level', default='INFO', type=str, 
                        help='Log Level.')
    parser.add_argument('--prof', dest='profile', type=str, 
                        default='Application-Profiler.xml',
                        help='Specify profile output file.')

def addEarlyStop (parser) :
    '''Setup common early stoppage parameters.'''
    import numpy as np
    parser.add_argument('--limit', dest='limit', type=int, default=2,
                        help='Number of runs between validation checks.')
    parser.add_argument('--stop', dest='stop', type=int, default=5,
                        help='Number of inferior validation checks to end.')
    parser.add_argument('--epoch', dest='epoch', type=float, default=np.inf,
                        help='Maximum number of runs per Layer/Network.')

def addSupDataParams (parser, base) :
    '''Setup common dataset parameters for supervised learning.'''
    parser.add_argument('--batch', dest='batchSize', type=int, default=100,
                        help='Batch size for training and test sets.')
    parser.add_argument('--holdout', dest='holdout', type=float, default=.05,
                        help='Percent of data to be held out for testing.')
    parser.add_argument('--base', dest='base', type=str, default='./' + base,
                        help='Base name of the network output and temp files.')
    parser.add_argument('--syn', dest='synapse', type=str, default=None,
                        help='Load from a previously saved network.')
    parser.add_argument('data', help='Directory or pkl.gz file for the ' +
                                     'training and test sets.')

def addUnsupDataParams (parser, base) :
    '''Setup common dataset parameters for unsupervised learning.'''
    addSupDataParams(parser, base)
    parser.add_argument('--target', dest='targetDir', type=str, required=True,
                        help='Directory with target data to match.')

def addSupConvolutionalParams(parser) :
    '''Setup common ConvolutionalLayer options.'''
    parser.add_argument('--kernel', dest='kernel', type=int, nargs='+',
                        default=[],
                        help='Number of kernels on Convolutional Layers.')
    parser.add_argument('--kernelSize', dest='kernelSize', type=int, nargs='+',
                        default=[],
                        help='Size of kernels on Convolutional Layers.')
    parser.add_argument('--downsample', dest='downsample', type=int, nargs='+',
                        default=[],
                        help='Downsample factor on Convolutional Layers.')
    parser.add_argument('--learnC', dest='learnC', type=float, nargs='+',
                        default=[],
                        help='Rate of learning on Convolutional Layers.')
    parser.add_argument('--momentumC', dest='momentumC', type=float, nargs='+',
                        default=[],
                        help='Rate of momentum on Convolutional Layers.')
    parser.add_argument('--dropoutC', dest='dropoutC', type=float, nargs='+',
                        default=[],
                        help='Dropout amount on Convolutional Layers.')
    parser.add_argument('--regTypeC', dest='regTypeC', type=str, 
                        default='L2',
                        help='Type of regularization on Convolutional Layers.')
    parser.add_argument('--regValueC', dest='regValueC', type=float, 
                        default=.00001,
                        help='Rate of regularization on Convolutional Layers.')

def addUnsupConvolutionalParams(parser) :
    '''Setup common ConvolutionalAE options.'''
    addSupConvolutionalParams(parser)
    parser.add_argument('--sparseC', dest='sparseC', type=bool, nargs='+',
                        default=[],
                        help='Force the output to be sparse for stronger '
                             'pattern extraction on Convolutional Layers.')

def addSupContiguousParams(parser) :
    '''Setup common ContiguousLayer options.'''
    parser.add_argument('--neuron', dest='neuron', type=int, nargs='+',
                        default=[500, 300, 100],
                        help='Number of neurons on Fully-Connected Layers.')
    parser.add_argument('--learnF', dest='learnF', type=float, nargs='+',
                        default=[.02, .02, .02],
                        help='Rate of learning on Fully-Connected Layers.')
    parser.add_argument('--momentumF', dest='momentumF', type=float, nargs='+',
                        default=[.2, .2, .2],
                        help='Rate of momentum on Fully-Connected Layers.')
    parser.add_argument('--dropoutF', dest='dropoutF', type=float, nargs='+',
                        default=[0.5, 0.5, 1],
                        help='Dropout amount on Fully-Connected Layer.')
    parser.add_argument('--regTypeF', dest='regTypeF', type=str, 
                        default='L2', help='Type of regularization on ' \
                                           'Fully-Connected Layers.')
    parser.add_argument('--regValueF', dest='regValueF', type=float, 
                        default=.00001, help='Rate of regularization on ' \
                                             'Fully-Connected Layers.')

def addUnsupContiguousParams(parser) :
    '''Setup common ContiguousAE options.'''
    addSupContiguousParams(parser)
    parser.add_argument('--sparseF', dest='sparseF', type=bool, nargs='+',
                        default=[],
                        help='Force the output to be sparse for stronger '
                             'pattern extraction on Fully-Connected Layers.')

def setupLogging (options, appName) :
    '''Grab the logger and parser from the options.'''
    import logging
    from builder.profiler import Profiler

    logName = appName + ': ' + options.data

    # setup the logger
    log = logging.getLogger(logName)
    log.setLevel(options.level.upper())
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    stream = logging.StreamHandler()
    stream.setLevel(options.level.upper())
    stream.setFormatter(formatter)
    log.addHandler(stream)

    # attach it to a file -- if requested
    if options.logfile is not None :
        logFile = logging.FileHandler(options.logfile)
        logFile.setLevel(options.level.upper())
        logFile.setFormatter(formatter)
        log.addHandler(logFile)

    # setup the profiler
    prof = Profiler(log=log, name=logName, profFile=options.profile)

    return log, prof
