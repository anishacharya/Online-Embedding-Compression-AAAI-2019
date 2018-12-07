# Training Hyperparameters
import os
import math
os.environ["CUDA_VISIBLE_DEVICES"]="0"

'''
Author: Anish Acharya <achanish@amazon.com>
'''
#export CUDA_VISIBLE_DEVICES=1

## Network Config
dropout = 'true'
DROPOUT_PROB = 0.5
nl= 'relu'  #Options: relu, tanh
batch_norm = 'false'
REGULARIZATION='false'
REGULARIZATION_FRACTION=0.0001

# Epoch Configs
STARTING_EPOCH= 0
EPOCH = 3 

# Batch Scheduling Configs
BATCH_SIZE = 50
Batch_Schedule = 'false'
MAX_BATCH_SIZE = 1000
boost = 2
#batch_freq = 2

#initialization Configs
mean = 0
std = 0.5 # Std Dev of Normal initialization

# Optimization Configs
optimize = 'ADAM' #'ADAGRAD' #'ADAM'#'ADAM' #'ADAGRAD'#'ADAGRAD' # options: SGD / ADAM/ RMSProp / ADAGRAD
decay = 0.1
freq  = 1

LR_Schedule = 'cyclic'
LR_high_initial = 0.001
LR_high = LR_high_initial
LR_low =  LR_high_initial* math.pow(decay , 5)
#LR_high = 0.005
#LR_low =  LR_high* math.pow(decay , 5)
step_size = 1000
#lr_boost=100
## At Each 'freq' Epoch Sets the LR = LR*decay
## for Cyclic schedule sets the high LR to LR*decay ,
## set decay = 1 if you want non-decayed LR/ non-decayed Cycles

# For ADAM
beta1=0.9
beta2=0.999
epsilon=0.01
