import numpy as np
import theano as theano
import theano.tensor as T

def leaky_relu(z , alpha = 0):
    if alpha == 1:
    	return z
    else:
    	return T.switch(z < 0, alpha * z, z)

def sigmoid(z):
    return T.nnet.nnet.sigmoid(z)

def linear(z):
    return z

def hyperbolic_tangent(z):
    return T.tanh(z)

def softmax(z):
    return T.nnet.softmax(z)[0]

