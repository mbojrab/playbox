
def softmaxAction(input, temp=1.) :
    '''This is a softmax function for action selection. This is typically used
       for performance learning, however it also aids in distillation.
       temp == 1 : regular Softmax return
       temp >  1 : softer Softmax return
    '''
    import theano.tensor as t
    eIn = t.exp(input / temp)
    return eIn * (1. / eIn.sum(axis=0))
