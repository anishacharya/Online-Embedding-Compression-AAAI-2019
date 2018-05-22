import numpy as np
import tensorflow as tf

def relu(z):
    if z < 0:
        return 0
    else:
        return z

def leaky_relu(z , alpha = 0):
    if z < 0:
        return alpha * z
    else:
	return z

def sigmoid(z):
    return tf.sigmoid(z)

def linear(z):
    return z

def hyperbolic_tangent(z):
    return tf.tanh(z)

def softmax(z):
    return tf.softmax(z)[0]
