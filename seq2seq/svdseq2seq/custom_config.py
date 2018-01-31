import sys
import os
import numpy as np

datasets_dir = os.getcwd() + '/' + '../data/'


netType = "svdMLP" #"svdseq2seq" # Options: seq2seq , svdseq2seq ,  + To Add Attention Options
n_in = 28*28            # i/p dim 
n_out = 10              # o/p dim
n_h = 128		# hidden dim
n_r = 16		# num of HouseHolder Reflectors (For svd Formulation)
n_layers=10		# num of Hidden Layers
m = 0.1			# Singular Margin 

num_epoch = 10 #50			# Number of Epoch
batchsize = 1000			# Instance Per Batch
validation_int = 100			# Validation Interval
learning_rate = 0.0001			# Learning Rate SGD
dropout = 0.1				# DropOut Rate
sig_mean = 1.0    			# SigMoid Mean 
activation_Hidden = 'leaky_relu'	# sigmoid 
activation_ooutput = 'softmax'		# Final Layer Activation : Softmax: Classification, Sigmoid: Regression 


rng = np.random.RandomState(13)
 
