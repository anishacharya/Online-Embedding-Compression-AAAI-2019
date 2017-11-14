from theano import function, config, shared, tensor
import numpy as np
import theano
import time
from svdRNN import svdRNN
from oRNN import oRNN
from RNN import RNN
from LSTM import LSTM
from load import mnist, UCR, Adding_task
from load import UCR50words
import sys

def usageAndExit():
	print "Useage: python run_rnn.py [Net]\n"
	sys.exit(0)

if len(sys.argv) < 2:
	usageAndExit()

trX, teX, trY, teY = Adding_task()
#trX, teX, trY, teY = mnist( onehot=False)

print "Training data: ", trX.shape, trY.shape
print "Test data: ", teX.shape, teY.shape

netType = sys.argv[1]
n_in = 2
n_out = 1
n_h = 128
n_r = 16
m = 0.01
sig_mean = 1.0

num_epoch = 100
batchsize = 10
validation_int = 1
learning_rate = 0.003
dropout_rate = 0.02

print "Running network: ", netType
print "Input dimension: ", n_in
print "Output dimension: ", n_out
print "Hidden dimension: ", n_h
print "Number of reflection vectors: ", n_r
print "Singualr margin: ", m
print "Hidden-to-hidden activation function: ", "leacky relu"
print "Output activation function: ", "softmax"
print "Instance per batch: ", batchsize
print "Validation interval: ", validation_int
print "Number of epoch: ", num_epoch
print "Learning rate: ", learning_rate
print "Mean of sigma: ", sig_mean
print "Dropout rate: ", dropout_rate


len_trX = np.array([x.shape[0]/n_in for x in trX], dtype=int)
len_teX = np.array([x.shape[0]/n_in for x in teX], dtype=int)
theano.config.optimizer='fast_run'

rng = np.random.RandomState(13)


if netType == "oRNN":
	net = oRNN(rng, n_in, n_out, n_h, n_r, obj='r')
elif netType == "RNN":
	net = RNN(rng, n_in, n_out, n_h, orth_init=True, dropout_rate = dropout_rate, obj='r')
elif netType == "LSTM":
	net = LSTM(rng, n_in, n_out, n_h, obj='r')
elif netType == "svdRNN":
	net = svdRNN(rng, n_in, n_out, n_h, n_r, margin = m, sig_mean = sig_mean, obj = 'r')
else:
	print "Net type err"


result = net.optimiser.train(rng, [trX.astype('f'),trY, len_trX],
                            valid_data =[teX.astype('f'), teY, len_teX],
                            instances_per_batch = batchsize,
                            valid_freq = validation_int,
                            lambda_ = learning_rate,
                            n_epoch = num_epoch,
                            verbose = 2)
