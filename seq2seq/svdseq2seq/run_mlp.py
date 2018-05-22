## External
import numpy as np
import time
import sys

## Internal
import custom_config
import load
#import activations
import utils

def usageAndExit():
        print "Useage: python run_mlp.py" 
        sys.exit(0)

if len(sys.argv) > 1:
        usageAndExit()

trX, teX, trY, teY = load.mnist(onehot=False) 

print "Training data: ", trX.shape, trY.shape
print "Test data: ", teX.shape, teY.shape
utils.print_config()


