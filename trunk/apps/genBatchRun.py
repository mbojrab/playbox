import sys
import argparse
from nn.profiler import setupLogging
import numpy as np
import os.path as osp


def buildParser(parser=argparse.ArgumentParser()):
    parser.add_argument('--script', type=str, default=None,
                        help='script file to generate stuff for')
    return parser


'''This is a simple batch generator for lenet5Trainer.py. All tweak-able values
   in lenet5Trainer have min, max and step here. This generates a dense matrix
   of processing parameters sets into a batch file for the appropriate OS.
'''
if __name__ == '__main__' :
    parser = buildParser(argparse.ArgumentParser())
    #parser.add_argument('--log', dest='logfile', type=str, default=None,
    #                    help='Specify log output file.')
    #parser.add_argument('--level', dest='level', default='INFO', type=str, 
    #                    help='Log Level.')
    #parser.add_argument('--learnC', dest='learnC', type=float, nargs='+', 
    #                    default=[.1,1,.25],
    #                    help='Rate of learning on Convolutional Layers.')
    #parser.add_argument('--learnF', dest='learnF', type=float, nargs='+', 
    #                    default=[.1,1,.25], 
    #                    help='Rate of learning on Fully-Connected Layers.')
    #parser.add_argument('--momentum', dest='momentum', type=float, nargs='+', 
    #                    default=[.1,1,.4],
    #                    help='Momentum rate all layers.')
    #parser.add_argument('--dropout', dest='dropout', type=bool,
    #                    nargs='+', default=[False, True],
    #                    help='Enable dropout throughout the network. Dropout '\
    #                         'percentages are based on optimal reported '\
    #                         'results. NOTE: Networks using dropout need to '\
    #                         'increase both neural breadth and learning rates')
    #parser.add_argument('--kernel', dest='kernel', type=int, nargs='+', 
    #                    default=[20,80,20], 
    #                    help='Number of Convolutional Kernels in each Layer.')
    #parser.add_argument('--neuron', dest='neuron', type=int, nargs='+', 
    #                    default=[200,500,150],
    #                    help='Number of Neurons in Hidden Layer.')
    #parser.add_argument('--limit', dest='limit', type=int, nargs='+', 
    #                    default=[2],
    #                    help='Number of runs between validation checks.')
    #parser.add_argument('--stop', dest='stop', type=int, nargs='+', 
    #                    default=[20],
    #                    help='Number of inferior validation checks to end.')
    #parser.add_argument('--batch', dest='batchSize', type=int, nargs='+', 
    #                    default=[10,60,20],
    #                    help='Batch size for training and test sets.')
    #parser.add_argument('--syn', dest='synapse', type=str, default=None,
    #                    help='Load from a previously saved network.')
    #parser.add_argument('data', type=str, default=None,
    #                    help='Directory or pkl.gz file for the training ' + \
    #                         'and test sets')
    #parser.add_argument('script', type=str, default=None,
    #                    help='script file to generate stuff for')
    options, rest = parser.parse_known_args()

    appmodulename = osp.splitext(options.script)[0]

    appmodule = __import__(appmodulename)

    parser = appmodule.buildParser(buildParser())
    options = parser.parse_args()

    #import pdb; pdb.set_trace()

    excludes = set(['script'])

    optiondict = vars(options)

    OPT_STR, OPT_DEST, OPT_VALUE, OPT_TYPE = range(0, 4)
    optionmapping = [(p.option_strings, p.dest, optiondict[p.dest], p.type)
                     for p in parser._actions
                     if p.dest in optiondict and
                        p.dest not in excludes and
                        optiondict[p.dest] is not None]
    # sort by the option strings
    optionmapping.sort(key=lambda tup: tup[OPT_STR])

    # setup the logger
    logName = 'batchGen: ' + str(options.data)
    log = setupLogging(logName, options.level, options.logfile)

    def genSteps(args) :
        '''Generate the range specified by the user.'''
        #if isinstance(args, str):  ## a string arg might have len == 3
        #    return args
        return np.arange(args[0], args[1], args[2]) if len(args) == 3 else args

    def permute(*args) :
        '''Generate all possible permutations of the parameters.'''
        import itertools
        # if args are passed as a list, unwrap them
        a = args[0] if len(args) == 1 else args
        paramSets = [genSteps(x) for x in a]
        return list(itertools.product(*paramSets))


    # get the permutations in the same order as for the arguments
    permutable = [tup for tup in optionmapping if
                  len(tup[OPT_STR]) > 0 and tup[OPT_TYPE] is not str
                  and hasattr(tup[OPT_VALUE], '__len__')]

    # put the stuff that shouldn't be permuted separately
    impermutable = [tup for tup in optionmapping if
                    len(tup[OPT_STR]) < 1 or tup[OPT_TYPE] is str
                    or not hasattr(tup[OPT_VALUE], '__len__')]

    permutations = permute([tup[OPT_VALUE] for tup in permutable])

    filename = 'batch' + ('.bat' if sys.platform == 'win32' else '.sh')
    with open(filename, 'w') as f :
        for perm in permutations :
            perm = [str(x) for x in perm]            
            cmd = 'python {0}'.format(options.script)
            for optmap, p in zip(permutable, perm):
                cmd += ' {0} {1}'.format(optmap[OPT_STR][0], p)


            # TODO: may need to sort the mandatory args 
            # to the end
            for optmap in impermutable:
                arg = optmap[OPT_VALUE]
                if len(optmap[0]) > 0:
                    cmd += ' {0} {1}'.format(optmap[OPT_STR][0], arg)
                else:
                    cmd += ' {0} '.format(arg)

            f.write(cmd + '\n')
