from theano import function, config, shared, tensor
import numpy as np
import theano
import time
from MLP import MLP
from svdMLP import svdMLP
from load import mnist
import sys


netType = sys.argv[1]
n_in = 28*28; n_h = [112, 56, 56, 28, 28]; n_out = 10
n_ri =  [ 196, 56, 28, 28, 14, 14]
n_ro =  [ 56, 28, 28, 14, 14, 10]
m = 0.1


num_epoch = 50
batchsize = 1
validation_int = 10
learning_rate = 0.0001
dropout_rate=0.1

print "Running network: ", netType
print "Input dimension: ", n_in
print "Output dimension: ", n_out
print "Hidden layers: ", len(n_h)+1
print "Hidden dimension: ", n_h
print "Number of input reflection vectors: ", n_ri
print "Number of output reflection vectors: ", n_ro
print "Singualr margin: ", m
print "Hidden-to-hidden activation function: ", "leacky relu"
print "Output activation function: ", "softmax"
print "Instance per batch: ", batchsize
print "Validation interval: ", validation_int
print "Number of epoch: ", num_epoch
print "Learning rate: ", learning_rate
print "Dropout rate: ", dropout_rate

theano.config.optimizer='fast_compile'
theano.config.exception_verbosity='high'

trX, teX, trY, teY = mnist(onehot=False)

print "Training data: ", trX.shape, trY.shape
print "Test data: ", teX.shape, teY.shape


rng = np.random.RandomState(724)


if netType == "MLP":
	net = MLP(rng, n_in, n_out, n_h, dropout_rate = dropout_rate)
elif netType == "svdMLP":
    net = svdMLP(rng, n_in, n_out, n_h, n_ri, n_ro, margin=m, dropout_rate = dropout_rate)
else:
	print "Net type err"

result = net.optimiser.train(rng, [trX.astype('f'),trY],
                            valid_data = [teX.astype('f'), teY],
                            instances_per_batch = batchsize,
                            valid_freq = validation_int,
                            n_epoch = num_epoch,
                            lambda_ = learning_rate,
                            verbose = 2)
