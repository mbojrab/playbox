import sys
import argparse
from nn.profiler import setupLogging
import numpy as np

'''This is a simple batch generator for lenet5Trainer.py. All tweak-able values
   in lenet5Trainer have min, max and step here. This generates a dense matrix
   of processing parameters sets into a batch file for the appropriate OS.
'''
if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', dest='logfile', type=str, default=None,
                        help='Specify log output file.')
    parser.add_argument('--level', dest='level', default='INFO', type=str, 
                        help='Log Level.')
    parser.add_argument('--learnC', dest='learnC', type=list, nargs=3, 
                        default=[.01,.1,.03],
                        help='Rate of learning on Convolutional Layers.')
    parser.add_argument('--learnF', dest='learnF', type=list, nargs=3, 
                        default=[.01,.1,.03], 
                        help='Rate of learning on Fully-Connected Layers.')
    parser.add_argument('--momentum', dest='momentum', type=list, nargs=3, 
                        default=[.1,.9,.25],
                        help='Momentum rate all layers.')
    parser.add_argument('--dropout', dest='dropout', type=list,
                        nargs=2, default=[False, True],
                        help='Enable dropout throughout the network. Dropout '\
                             'percentages are based on optimal reported '\
                             'results. NOTE: Networks using dropout need to '\
                             'increase both neural breadth and learning rates')
    parser.add_argument('--kernel', dest='kernel', type=list, nargs=3, 
                        default=[20,80,20], 
                        help='Number of Convolutional Kernels in each Layer.')
    parser.add_argument('--neuron', dest='neuron', type=list, nargs=3, 
                        default=[200,500,150],
                        help='Number of Neurons in Hidden Layer.')
    parser.add_argument('--limit', dest='limit', type=list, nargs=3, 
                        default=[1,3,2],
                        help='Number of runs between validation checks.')
    parser.add_argument('--stop', dest='stop', type=list, nargs=3, 
                        default=[5,20,10],
                        help='Number of inferior validation checks to end.')
    parser.add_argument('--batch', dest='batchSize', type=list, nargs=3, 
                        default=[5,100,50],
                        help='Batch size for training and test sets.')
    parser.add_argument('--syn', dest='synapse', type=str, default=None,
                        help='Load from a previously saved network.')
    parser.add_argument('--data', type=str, default=None,
                        help='Directory or pkl.gz file for the training ' + \
                             'and test sets')
    options = parser.parse_args()

    # setup the logger
    logName = 'batchGen: ' + str(options.data)
    log = setupLogging(logName, options.level, options.logfile)

    def genSteps(args) :
        '''Generate the range specified by the user.'''
        return np.arange(args[0], args[1], args[2]) if len(args) == 3 else args

    def permute(learnC, learnF, momentum, dropout,
                kernel, neuron, limit, stop, batch) :
        '''Generate all possible permutations of the parameters.'''
        import itertools
        params = [learnC, learnF, momentum, dropout,
                  kernel, neuron, limit, stop, batch]
        paramSets = [genSteps(x) for x in params]
        return list(itertools.product(*paramSets))

    permutations = permute(options.learnC, options.learnF, options.momentum,
                           options.dropout, options.kernel, options.neuron,
                           options.limit, options.stop, options.batchSize)

    filename = 'batch' + '.bat' if sys.platform == 'win32' else '.sh'
    with open(filename, 'w') as f :
        for perm in permutations :
            perm = [str(x) for x in perm]            
            cmd = 'python .\lenet5Trainer.py'
            cmd += ' --learnC '   + perm[0]
            cmd += ' --learnF '   + perm[1]
            cmd += ' --momentum ' + perm[2]
            cmd += ' --dropout '  + perm[3]
            cmd += ' --kernel '   + perm[4]
            cmd += ' --neuron '   + perm[5]
            cmd += ' --limit '    + perm[6]
            cmd += ' --stop '     + perm[7]
            cmd += ' --batch '    + perm[8]
            
            if options.synapse is not None :
                cmd += ' --syn ' + options.synapse
            if options.data is not None :
                cmd += ' ' + options.data

            f.write(cmd + '\n')