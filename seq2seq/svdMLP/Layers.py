import time
import numpy as np
import theano as theano
import theano.tensor as T
from svd_tensor_ops import svd_H_wy
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

def sigmoid(z):
    return T.nnet.nnet.sigmoid(z)

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, predict_input = None, W=None, b=None,
                 activation=T.tanh, dropout_rate = 0, nametag=''):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        if predict_input is None:
            predict_input = input
        self.input = input
        self.predict_input = predict_input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W'+nametag, borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b'+nametag, borrow=True)

        self.W = W
        self.b = b

        output = activation(T.dot(input, self.W) + self.b)
        predict_output = activation(T.dot(predict_input, self.W) + self.b)
        if dropout_rate > 0:
            #np.random.seed(int(time.time()))
            #mask = np.random.binomial(np.ones(n_out, dtype=int),1.0-dropout_rate)
            srng = RandomStreams(rng.randint(999999))
            mask = srng.binomial(n=1, p=1.0-dropout_rate, size=output.shape, dtype=theano.config.floatX)
            self.output = output * mask
            self.predict_output = (1.0-dropout_rate) * predict_output
        else:
            self.output = output
            self.predict_output = predict_output
        # parameters of the model
        self.params = [self.W, self.b]

class svd_HiddenLayer(object):
    def __init__(self, rng, input, n_h, n_r, predict_input = None, margin=0.01,
                 activation=T.tanh, dropout_rate = 0,nametag=''):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type n_h: int
        :param n_h: hidden dimension

        :type n_r: int
        :param n_r: number of Householder reflectors

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        if predict_input is None:
            predict_input = input
        self.input = input
        self.predict_input = predict_input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.

        U_ = np.tril(rng.normal(0, 0.01, (n_h, n_r)))
        norms_U_ = np.linalg.norm(U_, axis=0)
        U_ = 1. / norms_U_ * U_

        V_ = np.tril(rng.normal(0, 0.01, (n_h, n_r)))
        norms_V_ = np.linalg.norm(V_, axis=0)
        V_ = 1. / norms_V_ * V_

        #Sig_ = np.ones( n_h)
        P_ = np.random.randn(n_h)

        U = theano.shared(name='U'+nametag, value=U_.astype(theano.config.floatX))
       	V = theano.shared(name='V'+nametag, value=V_.astype(theano.config.floatX))
        P = theano.shared(name='P'+nametag, value=P_.astype(theano.config.floatX))

        b_values = np.zeros((n_h,), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, name='b'+nametag, borrow=True)

        self.U = U
        self.V = V
        self.P = P
        self.b = b

        Sig = 2*margin*(sigmoid(P)-0.5) + 1.0
        if n_h != n_r:
            lin_output = svd_H_wy(self.U, self.V, Sig, input)  + self.b
            predict_lin_output = svd_H_wy(self.U, self.V, Sig, predict_input)  + self.b
        else:
            Hu1SigHv1 = T.set_subtensor(Sig[-1], Sig[-1] * U[-1,-1] * V[-1,-1])
            lin_output = svd_H_wy(self.U[:,:-1], self.V[:,:-1], Hu1SigHv1, input)  + self.b
            predict_lin_output = svd_H_wy(self.U[:,:-1], self.V[:,:-1], Hu1SigHv1, predict_input)  + self.b

        output = activation(lin_output)
        predict_output = activation(predict_lin_output)
        if dropout_rate > 0:
            np.random.seed(int(time.time()))
            mask = np.random.binomial(np.ones(n_h, dtype=int),1.0-dropout_rate)
            self.output = output * T.cast(mask, theano.config.floatX)
            self.predict_output = (1.0-dropout_rate) * predict_output
        else:
            self.output = output
            self.predict_output = predict_output
        # parameters of the model
        self.params = [self.U, self.V, self.P, self.b]

class svd_HiddenLayer2(object):
    def __init__(self, rng, input, n_in, n_out, n_ri, n_ro, predict_input = None, margin=0.01,
                 activation=T.tanh, dropout_rate = 0,nametag=''):
        """
        NOTE : The nonlinearity used here is tanh

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type n_ri: int
        :param n_out: number of Householder reflectors for input dimension (V)

        :type n_ro: int
        :param n_out: number of Householder reflectors for output dimension (U)

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        if predict_input is None:
            predict_input = input
        self.input = input
        self.predict_input = predict_input

        assert(n_ri<=n_in)
        assert(n_ro<=n_out)

        U_ = np.tril(rng.normal(0, 0.01, (n_out, n_ro)))
        norms_U_ = np.linalg.norm(U_, axis=0)
        U_ = 1. / norms_U_ * U_

        V_ = np.tril(rng.normal(0, 0.01, (n_in, n_ri)))
        norms_V_ = np.linalg.norm(V_, axis=0)
        V_ = 1. / norms_V_ * V_

        P_ = np.random.randn(min(n_in,n_out))

        U = theano.shared(name='U'+nametag, value=U_.astype(theano.config.floatX))
       	V = theano.shared(name='V'+nametag, value=V_.astype(theano.config.floatX))
        P = theano.shared(name='P'+nametag, value=P_.astype(theano.config.floatX))

        b_values = np.zeros((n_out,), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, name='b'+nametag, borrow=True)

        self.U = U
        self.V = V
        self.P = P
        self.b = b

        Sig = 2*margin*(sigmoid(self.P)-0.5) + 1.0
        if n_in == n_ri and n_out == n_ro:
            Hu1SigHv1 = T.set_subtensor(Sig[-1], Sig[-1] * U[-1,-1] * V[-1,-1])
            lin_output = rect_svd_H_wy(self.U[:,:-1], self.V[:,:-1], Hu1SigHv1, input)  + self.b
            predict_lin_output = rect_svd_H_wy(self.U[:,:-1], self.V[:,:-1], Hu1SigHv1, predict_input)  + self.b
        elif n_in == n_ri and n_out != n_ro:
            SigHv1 = T.set_subtensor(Sig[-1], Sig[-1] *  V[-1,-1] )
            lin_output = rect_svd_H_wy(self.U[:,:-1], self.V[:,:-1], SigHv1, input)  + self.b
            predict_lin_output = rect_svd_H_wy(self.U[:,:-1], self.V[:,:-1], SigHv1, predict_input)  + self.b
        elif n_in != n_ri and n_out == n_ro:
            Hu1Sig = T.set_subtensor(Sig[-1], Sig[-1] * U[-1,-1] )
            lin_output = rect_svd_H_wy(self.U[:,:-1], self.V[:,:-1], Hu1Sig, input)  + self.b
            predict_lin_output = rect_svd_H_wy(self.U[:,:-1], self.V[:,:-1], Hu1Sig, predict_input)  + self.b
        else:
            lin_output = rect_svd_H_wy(self.U, self.V, Sig, input)  + self.b
            predict_lin_output = rect_svd_H_wy(self.U, self.V, Sig, predict_input)  + self.b

        output = activation(lin_output)
        predict_output = activation(predict_lin_output)
        if dropout_rate > 0:
            np.random.seed(int(time.time()))
            mask = np.random.binomial(np.ones(n_out, dtype=int),1-dropout_rate)
            self.output = output * T.cast(mask, theano.config.floatX)
            self.predict_output = (1.0-dropout_rate) * predict_output
        else:
            self.output = output
            self.predict_output = predict_output

        # parameters of the model
        self.params = [self.U, self.V, self.P, self.b]

class ResNetLayer(object):
    def __init__(self, rng, input, n_h, predict_input = None,
                 activation=T.tanh, dropout_rate = 0,nametag=''):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        if predict_input is None:
            predict_input = input
        self.input = input
        self.predict_input = predict_input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        W1_values = np.asarray(
            rng.uniform(
                low=-np.sqrt(6. / (n_h + n_h)),
                high=np.sqrt(6. / (n_h + n_h)),
                size=(n_h, n_h)
            ),
            dtype=theano.config.floatX
        )

        W1 = theano.shared(value=W1_values, name='W1_'+nametag, borrow=True)

        W2_values = np.asarray(
            rng.uniform(
                low=-np.sqrt(6. / (n_h + n_h)),
                high=np.sqrt(6. / (n_h + n_h)),
                size=(n_h, n_h)
            ),
            dtype=theano.config.floatX
        )

        W2 = theano.shared(value=W2_values, name='W2_'+nametag, borrow=True)

        b1_values = np.zeros((n_h,), dtype=theano.config.floatX)
        b1 = theano.shared(value=b1_values, name='b1_'+nametag, borrow=True)

        b2_values = np.zeros((n_h,), dtype=theano.config.floatX)
        b2 = theano.shared(value=b2_values, name='b2_'+nametag, borrow=True)

        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2

        inner = activation(T.dot(input, self.W1) + self.b1)
        predict_inner = activation(T.dot(predict_input, self.W1) + self.b1)
        output = activation(T.dot(inner, self.W2) + self.b2 + input)
        predict_output = activation(T.dot(predict_inner, self.W2) + self.b2 + predict_input)

        if dropout_rate > 0:
            np.random.seed(int(time.time()))
            mask = np.random.binomial(np.ones(n_h, dtype=int),1-dropout_rate)
            self.output = output * T.cast(mask, theano.config.floatX)
            self.predict_output = (1.0-dropout_rate) * predict_output
        else:
            self.output = output
            self.predict_output = predict_output
        # parameters of the model
        self.params = [self.W1, self.b1, self.W2, self.b2]



class svdResNetLayer(object):
    def __init__(self, rng, input, n_h, n_r, predict_input = None,  margin=0.01,
                 activation=T.tanh, dropout_rate = 0, nametag=''):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        if predict_input is None:
            predict_input = input
        self.input = input
        self.predict_input = predict_input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        U1_ = np.tril(rng.normal(0, 0.01, (n_h, n_r)))
        norms_U1_ = np.linalg.norm(U1_, axis=0)
        U1_ = 1. / norms_U1_ * U1_

        V1_ = np.tril(rng.normal(0, 0.01, (n_h, n_r)))
        norms_V1_ = np.linalg.norm(V1_, axis=0)
        V1_ = 1. / norms_V1_ * V1_

        #Sig_ = np.ones( n_h)
        P1_ = np.random.randn(n_h)

        U1 = theano.shared(name='U1'+nametag, value=U1_.astype(theano.config.floatX))
       	V1 = theano.shared(name='V1'+nametag, value=V1_.astype(theano.config.floatX))
        P1 = theano.shared(name='P1'+nametag, value=P1_.astype(theano.config.floatX))

        U2_ = np.tril(rng.normal(0, 0.01, (n_h, n_r)))
        norms_U2_ = np.linalg.norm(U2_, axis=0)
        U2_ = 1. / norms_U2_ * U2_

        V2_ = np.tril(rng.normal(0, 0.01, (n_h, n_r)))
        norms_V2_ = np.linalg.norm(V2_, axis=0)
        V2_ = 1. / norms_V2_ * V2_

        #Sig_ = np.ones( n_h)
        P2_ = np.random.randn(n_h)

        U2 = theano.shared(name='U2'+nametag, value=U2_.astype(theano.config.floatX))
       	V2 = theano.shared(name='V2'+nametag, value=V2_.astype(theano.config.floatX))
        P2 = theano.shared(name='P2'+nametag, value=P2_.astype(theano.config.floatX))

        b1_values = np.zeros((n_h,), dtype=theano.config.floatX)
        b1 = theano.shared(value=b1_values, name='b1_'+nametag, borrow=True)

        b2_values = np.zeros((n_h,), dtype=theano.config.floatX)
        b2 = theano.shared(value=b2_values, name='b2_'+nametag, borrow=True)

        self.U1 = U1
        self.V1 = V1
        self.P1 = P1
        self.b1 = b1

        self.U2 = U2
        self.V2 = V2
        self.P2 = P2
        self.b2 = b2


        Sig1 = 2*margin*(sigmoid(P1)-0.5) + 0.0
        Sig2 = 2*margin*(sigmoid(P2)-0.5) + 0.0

        if n_h != n_r:
            inner = activation(svd_H_wy(self.U1, self.V1, Sig1, input)  + self.b1)
            predict_inner = activation(svd_H_wy(self.U1, self.V1, Sig1, predict_input)  + self.b1)
            outer = activation(svd_H_wy(self.U2, self.V2, Sig2, inner)  + self.b2 + input)
            predict_outer = activation(svd_H_wy(self.U2, self.V2, Sig2, predict_inner)  + self.b2 + predict_input)
        else:
            Hu1SigHv1_1 = T.set_subtensor(Sig1[-1], Sig1[-1] * U1[-1,-1] * V1[-1,-1])
            inner = activation(svd_H_wy(self.U1[:,:-1], self.V1[:,:-1], Hu1SigHv1_1, input)  + self.b1)
            predict_inner = activation(svd_H_wy(self.U1[:,:-1], self.V1[:,:-1], Hu1SigHv1_1, predict_input)  + self.b1)
            Hu1SigHv1_2 = T.set_subtensor(Sig2[-1], Sig2[-1] * U2[-1,-1] * V2[-1,-1])
            outer = activation( svd_H_wy(self.U2[:,:-1], self.V2[:,:-1], Hu1SigHv1_2, inner)  + self.b2 + input)
            predict_outer = activation( svd_H_wy(self.U2[:,:-1], self.V2[:,:-1], Hu1SigHv1_2, predict_inner)  + self.b2 + predict_input)

        output = outer
        predict_output = predict_outer
        if dropout_rate > 0:
            np.random.seed(int(time.time()))
            mask = np.random.binomial(np.ones(n_h, dtype=int),1-dropout_rate)
            self.output = output * T.cast(mask, theano.config.floatX)
            self.predict_output = (1.0-dropout_rate) * predict_output
        else:
            self.output = output
            self.predict_output = predict_output
        # parameters of the model
        self.params = [self.U1, self.V1, self.P1, self.b1, self.U2, self.V2, self.P2, self.b2]


