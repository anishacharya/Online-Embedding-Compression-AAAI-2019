import numpy as np
import theano as theano
import theano.tensor as T
from svd_tensor_ops import svd_H_wy
from sgd import sgd_optimizer

def leaky_relu(z):
    return T.switch(z < 0, 0.1 * z, z)

def leaky_relu_0(z):
    return T.switch(z < 0, 0.0 * z, z)

def leaky_relu_1(z):
    return T.switch(z < 0, 0.1 * z, z)

def leaky_relu_3(z):
    return T.switch(z < 0, 0.3 * z, z)

def leaky_relu_5(z):
    return T.switch(z < 0, 0.5 * z, z)

def leaky_relu_7(z):
    return T.switch(z < 0, 0.7 * z, z)

def leaky_relu_9(z):
    return T.switch(z < 0, 0.9 * z, z)

def leaky_relu_10(z):
    return z

def sigmoid(z):
    return T.nnet.nnet.sigmoid(z)

def linear(z):
    return z

def hyperbolic_tangent(z):
    return T.tanh(z)

def softmax(z):
    return T.nnet.softmax(z)[0]

