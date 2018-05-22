import numpy as np
import custom_config
## This function creates a random Orthogonal matrix through HouseHolder Reflectors ###

def rvs(random_state,dim):
     H = np.eye(dim)
     D = np.ones((dim,))
     for n in range(1, dim):
         x = random_state.normal(size=(dim-n+1,))
         D[n-1] = np.sign(x[0])
         x[0] -= D[n-1]*np.sqrt((x*x).sum())
         # Householder transformation
         Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
         mat = np.eye(dim)
         mat[n-1:, n-1:] = Hx
         H = np.dot(H, mat)
         # Fix the last sign such that the determinant is 1
     D[-1] = (-1)**(1-(dim % 2))*D.prod()
     # Equivalent to np.dot(np.diag(D), H) but faster, apparently
     H = (D*H.T).T
     return H

def print_config():
	print "Running network: ", custom_config.netType
	print "Input dimension: ", custom_config.n_in
	print "Output dimension: ", custom_config.n_out
	print "Hidden layers: ", custom_config.n_layers
	print "Hidden dimension: ", custom_config.n_h
	print "Number of reflection vectors: ", custom_config.n_r
	print "Singualr margin: ", custom_config.m
	print "Hidden-to-hidden activation function: ", "leacky relu"
	print "Output activation function: ", "softmax"
	print "Instance per batch: ", custom_config.batchsize
	print "Validation interval: ", custom_config.validation_int
	print "Number of epoch: ", custom_config.num_epoch
	print "Learning rate: ", custom_config.learning_rate
	print "Droput rate: ", custom_config.dropout    	
