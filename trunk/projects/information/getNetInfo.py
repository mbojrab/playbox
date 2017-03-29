import argparse
from nn.net import ClassifierNetwork as Net
from nn.contiguousLayer import ContiguousLayer
from nn.convolutionalLayer import ConvolutionalLayer

from builder.args import addLoggingParams, addSynapseLoad, addDebuggingParams
from builder.profiler import setupLogging
from dataset.shared import getShape

'''This is a simple network in the topology of leNet5 the well-known
   MNIST dataset trainer from Yann LeCun. This is capable of training other
   datasets, however the sizing must be correct.
'''
if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    addLoggingParams(parser)
    addSynapseLoad(parser, multiLoad=False)
    addDebuggingParams(parser)
    options = parser.parse_args()

    # setup the logger
    log, prof = setupLogging(options, 'cnnTrainer')

    # create the network -- LeNet-5
    network = Net(filepath=options.synapse, prof=prof, debug=options.debug)

    log.info('Network Information: \n' + str(network))
