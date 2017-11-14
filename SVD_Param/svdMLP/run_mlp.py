from theano import function, config, shared, tensor
import numpy as np
import theano
import time
from MLP import MLP
from svdMLP import svdMLP
from ResNet import ResNet
from svdResNet import svdResNet
from load import mnist
import sys

def usageAndExit():
	print "Useage: python run_mlp.py [Net]\n	Net=svdMLP/MLP"
	sys.exit(0)

if len(sys.argv) < 2:
	usageAndExit()

netType = sys.argv[1]

trX, teX, trY, teY = mnist(onehot=False)

print "Training data: ", trX.shape, trY.shape
print "Test data: ", teX.shape, teY.shape

n_in = 28*28
n_out = 10
n_h = 128
n_r = 16
n_layers=10
m = 0.1

num_epoch = 50
batchsize = 1
validation_int = 100
learning_rate = 0.0001
dropout = 0.1


print "Running network: ", netType
print "Input dimension: ", n_in
print "Output dimension: ", n_out
print "Hidden layers: ", n_layers
print "Hidden dimension: ", n_h
print "Number of reflection vectors: ", n_r
print "Singualr margin: ", m
print "Hidden-to-hidden activation function: ", "leacky relu"
print "Output activation function: ", "softmax"
print "Instance per batch: ", batchsize
print "Validation interval: ", validation_int
print "Number of epoch: ", num_epoch
print "Learning rate: ", learning_rate
print "Droput rate: ", dropout

theano.config.optimizer='fast_run'



rng = np.random.RandomState(724)


if netType == "MLP":
	net = MLP(rng, n_in, n_out, n_h, n_layers, dropout_rate = dropout)
elif netType == "svdMLP":
    net = svdMLP(rng, n_in, n_out, n_h, n_r,  n_layers, margin=m, dropout_rate = dropout)
elif netType == "svdResNet":
    net = svdResNet(rng, n_in, n_out, n_h, n_r,  n_layers, margin=m, dropout_rate = dropout)
elif netType == "ResNet":
    net = ResNet(rng, n_in, n_out, n_h, n_layers/2, dropout_rate = dropout)
else:
	print "Net type err"

result = net.optimiser.train(rng, [trX.astype('f'),trY],
                           valid_data = [teX.astype('f'), teY],
                            instances_per_batch = batchsize,
                            valid_freq = validation_int,
                            n_epoch = num_epoch,
                            lambda_ = learning_rate,
                            verbose = 2)
