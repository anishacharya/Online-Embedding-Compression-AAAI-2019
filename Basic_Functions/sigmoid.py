### Sigmoid Unit
import numpy as np

def sigmoid_scalar_calculation(x):
    return (1/(1 + np.exp(-x)))

def sigmoid_grad_scalar_calculation(sig):
    return (sig * (1 - sig))

def sigmoid(x):    
    vectorized_sigmoid_unit = np.vectorize(sigmoid_scalar_calculation)    
    return vectorized_sigmoid_unit(x)

def sigmoid_grad(f):
    """
    Compute the gradient for the sigmoid function here. Note that
    for this implementation, the input f should be the sigmoid
    function value of your original input x. 
    """
    vectorized_sigmoid_grad_unit = np.vectorize(sigmoid_grad_scalar_calculation)
    return vectorized_sigmoid_grad_unit(f)
    #raise NotImplementedError

def test_sigmoid_basic():
    """
    Some simple tests to get you started. 
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    x = np.array([[1, 2], [-1, -2]])
    f = sigmoid(x)
    g = sigmoid_grad(f)
    print f
    assert np.amax(f - np.array([[0.73105858, 0.88079708], 
        [0.26894142, 0.11920292]])) <= 1e-6
    print g
    assert np.amax(g - np.array([[2.19661193, 0.10499359],
        [0.19661193, 0.10499359]])) <= 1e-6
    # print "You should verify these results!\n"
    
if __name__ == "__main__":
    test_sigmoid_basic();
