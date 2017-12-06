import numpy as np
import random

def softmax(x):
    if len(x.shape) > 1:
        x = x - np.amax(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        exp_sum_row = np.sum(exp_x, axis=1, keepdims=True)
        sigm_x = exp_x / exp_sum_row
    else:
        x = x - np.max(x)
        exp_x = np.exp(x)
        exp_sum_row = np.sum(exp_x)
        sigm_x = exp_x / exp_sum_row
        
    return sigm_x

def test_softmax():
    print "Running basic tests..."
    test1 = softmax(np.array([1,2]))
    print test1
    assert np.amax(np.fabs(test1 - np.array(
        [0.26894142,  0.73105858]))) <= 1e-6

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print test2
    assert np.amax(np.fabs(test2 - np.array(
        [[0.26894142, 0.73105858], [0.26894142, 0.73105858]]))) <= 1e-6

    test3 = softmax(np.array([[-1001,-1002]]))
    print test3
    assert np.amax(np.fabs(test3 - np.array(
        [0.73105858, 0.26894142]))) <= 1e-6

if __name__ == "__main__":
    test_softmax()