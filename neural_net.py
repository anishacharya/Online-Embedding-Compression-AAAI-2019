import numpy as np
import random
from softmax import softmax
from sigmoid import sigmoid, sigmoid_grad
from gradcheck import gradcheck_naive


def activate(x, W, b):
    return np.dot(x,W)+b
    
def forward_prop(X_in,W1,b1,W2,b2):
    X_in = activate(X_in,W1,b1)  # Select Row 0 i.e. Data Point 0  => 1*H 
    X_in = sigmoid(X_in)         # Compute the Output of the Hidden Nodes of the Layer => 1*H 
    X_in = activate(X_in,W2,b2)  # Computes next Layer (in this Case Final Layer)    
    return softmax(X_in)
    
def forward_backward_prop(data, labels, params, dimensions):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """
    ### Unpack network parameters (do not modify)
    ofs = 0 
    cost= 0 
    grad= 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    # raise NotImplementedError
    ### END YOUR CODE
    
    # For one Row in Data Matrix / One Data Point
    X_in = data[[0],:]           # Data Row 1 is the input to Layer 1 
    Y_hat = forward_prop(X_in,W1,b1,W2,b2)
    
    print Y_hat
    print np.sum(Y_hat) # Should be 1 as Softmax Applied - So cross Check 
    
    
    ### YOUR CODE HERE: backward propagation
    # raise NotImplementedError
    ### END YOUR CODE
    
    ### Stack gradients (do not modify)
#   grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), 
#   gradW2.flatten(), gradb2.flatten()))
    
    return cost, grad



def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using 
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10] # Dx,H,Dy
    
    data = np.random.randn(N, dimensions[0])   # each row will be a datum  ---> 20, 10D Data Points # => N * Dx Matrix
    labels = np.zeros((N, dimensions[2]))      # So, 10 Classes -> Y # ==> So a N * Dy Dim Matrix Label 
    for i in xrange(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1   # Randomly Selected a class for each point - Row Wise    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (dimensions[1] + 1) * dimensions[2], )
                                               # Params: (Dx + 1)*Dh + (Dh + 1)*Dy = 55 + 60 = 115 
#    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,dimensions), params)
    forward_backward_prop(data, labels, params,dimensions)
    
if __name__ == "__main__":
    sanity_check()
