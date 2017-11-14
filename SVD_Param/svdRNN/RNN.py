import numpy as np
import theano as theano
import theano.tensor as T
from tensor_ops import H_wy
from sgd import sgd_optimizer
import time
from activations import *


class RNN():
    def __init__(self, rng, n_in, n_out, n_h, f_act=leaky_relu, f_out=softmax, orth_init=True, dropout_rate=0, obj = 'c'):
        '''
        :param rng: Numpy RandomState
        :param n_in: Input dimension (int)
        :param n_out: Output dimension (int)
        :param n_h: Hidden dimension (int)
        :param f_act: Hidden-to-hidden activation function
        :param f_out: Output activation function
        :param orth_init: if true, the initialize transition matrix to be orthogonal (bool)
        :param dropout_rate: dropout rate (float)
        :param obj: objective type, 'c' for classification with cross entropy loss, 'r' for regression with MSE loss. (['c','r'])
        '''
        if orth_init:
            Whh_ = rvs(rng,n_h)
        else:
            Whh_ = rng.uniform(-np.sqrt(6. / (n_h + n_h)), np.sqrt(6. / (n_h + n_h )), (n_h, n_h))

        Whi_ = rng.uniform(-np.sqrt(6. / (n_in + n_h)), np.sqrt(6. / (n_in + n_h)), (n_h, n_in))
        bh_ = np.zeros( n_h)
        Woh_ = rng.uniform(-np.sqrt(6. / (n_out + n_h)), np.sqrt(6. / (n_h + n_out)), (n_out, n_h))
        bo_ = np.zeros(n_out)
        h0_ = rng.uniform(-np.sqrt(3. / (2. * n_h)), np.sqrt(3. / (2. * n_h)), n_h)

        # Theano: Created shared variables
        Whh = theano.shared(name='Whh', value = Whh_.astype(theano.config.floatX))
        Whi = theano.shared(name='Whi', value=Whi_.astype(theano.config.floatX))
        bh = theano.shared(name='bh', value=bh_.astype(theano.config.floatX))
        Woh = theano.shared(name='Woh', value=Woh_.astype(theano.config.floatX))
        bo = theano.shared(name='bo', value=bo_.astype(theano.config.floatX))
        h0 = theano.shared(name='h0', value=h0_.astype(theano.config.floatX))

        self.p = [Whh, Whi, Woh, bh, bo, h0]

        seq_len = T.iscalar('seq_len')
        self.seq_len = seq_len
        self.dropout_rate = dropout_rate
        self.x = T.vector()
        x_scan = T.reshape(self.x, [seq_len, n_in], ndim = 2)


        if dropout_rate > 0:
            np.random.seed(int(time.time()))
            # for training
            def masked_forward_prop_step(x_t, h_t_prev):
                h_t = f_act(Whi.dot(x_t) + Whh.dot( h_t_prev) + bh)
                o_t = Woh.dot(h_t) + bo
                mask = np.random.binomial(np.ones(n_h, dtype=int),1-dropout_rate)
                masked_h_t = h_t * T.cast(mask, theano.config.floatX)

                return [o_t, masked_h_t]
            # for testing
            def forward_prop_step(x_t, h_t_prev):
                h_t = f_act(Whi.dot(x_t) + Whh.dot( h_t_prev) + bh)
                o_t = Woh.dot(h_t) + bo
                h_t = (1.0-dropout_rate) * h_t
                return [o_t, h_t]

            [o_train, _], _ = theano.scan(
                masked_forward_prop_step,
                sequences=[x_scan],
                outputs_info=[None, h0],
                n_steps=seq_len
            )

            [o_test, _], _ = theano.scan(
                forward_prop_step,
                sequences=[x_scan],
                outputs_info=[None, h0],
                n_steps=seq_len
            )

        else:
            def forward_prop_step(x_t, h_t_prev):
                h_t = f_act(Whi.dot(x_t) + Whh.dot( h_t_prev) + bh)
                o_t = Woh.dot(h_t) + bo
                return [o_t, h_t]

            [o_train, _], _ = theano.scan(
                forward_prop_step,
                sequences=[x_scan],
                outputs_info=[None, h0],
                n_steps=seq_len
            )
            o_test = o_train


        if obj == 'c': # classification task
            self.y = T.bscalar('y')
            self.o_train = f_out(o_train[-1])
            self.o_test = f_out(o_test[-1])
            #obj function to compute grad, use dropout
            self.cost = T.nnet.categorical_crossentropy(self.o_train, T.eye(n_out)[self.y])
            #compute accuracy use average of dropout rate
            self.accuracy = T.switch(T.eq(T.argmax(self.o_test), self.y), 1., 0.)
            self.prediction = np.argmax(self.o_test)
        elif obj == 'r': # regression task
            self.y = T.dscalar('y')
            self.o_train = o_train[-1]
            self.o_test = o_test[-1]
            #obj function to compute grad, use dropout
            self.cost = (self.o_train[0] - self.y)**2
            #compute accuracy use average of dropout rate
            self.accuracy = (self.o_test[0] - self.y)**2
            self.prediction = self.o_test[0]


        _, self.Sigma, _ = T.nlinalg.SVD( full_matrices=1, compute_uv=1)(self.p[0])
        self.max_singular = T.max(self.Sigma)
        self.min_singular = T.min(self.Sigma)


        self.optimiser = sgd_optimizer(self, 'RNN')

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
