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

def addDebuggingParams (parser) :
    '''Setup common debugging options.'''
    parser.add_argument('--debug', dest='debug', default=False,
                        action='store_true',
                        help='Dump debugging information while processing.')

def addEarlyStop (parser) :
    '''Setup common early stoppage parameters.'''
    import numpy as np
    parser.add_argument('--limit', dest='limit', type=int, default=2,
                        help='Number of runs between validation checks.')
    parser.add_argument('--stop', dest='stop', type=int, default=5,
                        help='Number of inferior validation checks to end.')
    parser.add_argument('--epoch', dest='epoch', type=float, default=np.inf,
                        help='Maximum number of runs per Layer/Network.')

def addSynapseLoad(parser, multiLoad=False) :
    '''Setup parser for loading synapses from disk to initialize a network.'''
    if not multiLoad :
        parser.add_argument('--syn', dest='synapse', type=str, default=None,
                            help='Load from a previously saved network.')
    else :
        parser.add_argument('--syn', dest='synapse', type=str, default=[],
                            nargs='*', help='Load one or more saved networks.')

def addSupDataParams (parser, base, multiLoad=False) :
    '''Setup common dataset parameters for supervised learning.'''
    parser.add_argument('--batch', dest='batchSize', type=int, default=100,
                        help='Batch size for training and test sets.')
    parser.add_argument('--holdout', dest='holdout', type=float, default=.05,
                        help='Percent of data to be held out for testing.')
    parser.add_argument('--base', dest='base', type=str, default='./' + base,
                        help='Base name of the network output and temp files.')
    addSynapseLoad(parser, multiLoad=multiLoad)
    parser.add_argument('data', help='Directory or pkl.gz file for the ' +
                                     'training and test sets.')

def addUnsupDataParams (parser, base, multiLoad=False) :
    '''Setup common dataset parameters for unsupervised learning.'''
    addSupDataParams(parser, base, multiLoad)
    parser.add_argument('--target', dest='targetDir', type=str, required=True,
                        help='Directory with target data to match.')
    parser.add_argument('--maxTarget', dest='maxTarget', type=int, default=100,
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
                        default=[],
                        help='Number of neurons on Fully-Connected Layers.')
    parser.add_argument('--learnF', dest='learnF', type=float, nargs='+',
                        default=[],
                        help='Rate of learning on Fully-Connected Layers.')
    parser.add_argument('--momentumF', dest='momentumF', type=float, nargs='+',
                        default=[],
                        help='Rate of momentum on Fully-Connected Layers.')
    parser.add_argument('--dropoutF', dest='dropoutF', type=float, nargs='+',
                        default=[],
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
