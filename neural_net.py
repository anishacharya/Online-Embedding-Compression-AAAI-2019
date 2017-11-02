import numpy as np
import random
from softmax import softmax
from sigmoid import sigmoid, sigmoid_grad
from gradcheck import gradcheck_naive

def activate(x, W, b):
    return np.dot(x,W) + b #x.dot(W) + b
    
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
    ### Unpack network parameters 
    ofs = 0 
    
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    
    ### forward propagation           
    z1 = activate(data,W1,b1)
    h  = sigmoid(z1)    
    z2 = activate(h,W2,b2)
    Y_hat = softmax(z2)
    
    #Y_hat = forward_prop(data,W1,b1,W2,b2)   # Data Row 1 is the input to Layer 1
    #print Y_hat.shape
    #print Y_hat
    #print np.sum(Y_hat) # Should be 1 as Softmax Applied - So cross Check 
    
    #print labels * np.log(Y_hat)
    cost   = -np.sum( np.log(Y_hat) * labels )
    #print cost
    
    dJ_dz2 = Y_hat - labels
    gradW2 = h.T.dot(dJ_dz2) #np.dot(data.T, dJ_dz2)
    gradb2 = np.sum(dJ_dz2, axis = 0)    
    dJ_dh  = dJ_dz2.dot(W2.T)#np.dot(dJ_dz2, W2.T)     
    dJ_dz1 = dJ_dh * sigmoid_grad(h) #np.multiply(dJ_dh, sigmoid_grad(z1))
    dJ_dx  = dJ_dz1.dot(W1.T) #np.dot(dJ_dz1, W1.T)
    gradW1 = data.T.dot(dJ_dz1) #np.dot(data.T, dJ_dz1)
    gradb1 = np.sum(dJ_dz1, axis = 0)
    
    #print dJ_dz1.shape,dJ_dh.shape,dJ_dz2.shape,dJ_dx.shape   
    ### Stack gradients 
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), gradW2.flatten(), gradb2.flatten()))
    #print grad
    
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
                                               ## Params: (Dx + 1)*Dh + (Dh + 1)*Dy = 55 + 60 = 115 
    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,dimensions), params)
    #forward_backward_prop(data, labels, params,dimensions)
    
if __name__ == "__main__":
    sanity_check()