import numpy as np
import theano as theano
import theano.tensor as T
from svd_tensor_ops import svd_H_wy
from sgd import sgd_optimizer
from activations import *


class svdRNN():
    def __init__(self, rng, n_in, n_out, n_h, n_r, margin = 1.0, sig_mean = 1.0,  f_act=leaky_relu, f_out=softmax, obj = 'c'):
        '''
        :param rng: Numpy RandomState
        :param n_in: Input dimension (int)
        :param n_out: Output dimension (int)
        :param n_h: Hidden dimension (int)
        :param n_r: Number of reflection vectors (int)
        :param f_act: Hidden-to-hidden activation function
        :param f_out: Output activation function
        :param obj: objective type, 'c' for classification with cross entropy loss, 'r' for regression with MSE loss. (['c','r'])
        '''
        U_ = np.tril(rng.normal(0, 0.01, (n_h, n_r)))
        norms_U_ = np.linalg.norm(U_, axis=0)
        U_ = 1. / norms_U_ * U_

        V_ = np.tril(rng.normal(0, 0.01, (n_h, n_r)))
        norms_V_ = np.linalg.norm(V_, axis=0)
        V_ = 1. / norms_V_ * V_

        #Sig_ = np.ones( n_h)
        P_ = np.zeros(n_h)

        Whi_ = rng.uniform(-np.sqrt(6. / (n_in + n_h)), np.sqrt(6. / (n_in + n_h)), (n_h, n_in))
        bh_ = np.zeros( n_h)
        Woh_ = rng.uniform(-np.sqrt(6. / (n_out + n_h)), np.sqrt(6. / (n_h + n_out)), (n_out, n_h))
        bo_ = np.zeros(n_out)
        h0_ = rng.uniform(-np.sqrt(3. / (2. * n_h)), np.sqrt(3. / (2. * n_h)), n_h)

        # Theano: Created shared variables
        Whi = theano.shared(name='Whi', value=Whi_.astype(theano.config.floatX))
        U = theano.shared(name='U', value=U_.astype(theano.config.floatX))
       	V = theano.shared(name='V', value=V_.astype(theano.config.floatX))
       	#Sig = theano.shared(name='Sig', value=Sig_.astype(theano.config.floatX))
       	P = theano.shared(name='P', value=P_.astype(theano.config.floatX))
        bh = theano.shared(name='bh', value=bh_.astype(theano.config.floatX))
        Woh = theano.shared(name='Woh', value=Woh_.astype(theano.config.floatX))
        bo = theano.shared(name='bo', value=bo_.astype(theano.config.floatX))
        h0 = theano.shared(name='h0', value=h0_.astype(theano.config.floatX))

        #self.p = [U, V, Sig, Whi, Woh, bh, bo, h0]
        self.p = [U, V, P, Whi, Woh, bh, bo, h0]
        seq_len = T.iscalar('seq_len')
        self.seq_len = seq_len

        self.x = T.vector()
        #x_scan = T.shape_padright(self.x)
        x_scan = T.reshape(self.x, [seq_len, n_in], ndim = 2)
        if n_h != n_r:  # Number of reflection vectors is less than the hidden dimension
            def forward_prop_step(x_t, h_t_prev):
                Sig = 2*margin*(sigmoid(P)-0.5) + sig_mean
                h_t = f_act(Whi.dot(x_t) + svd_H_wy(U,V,Sig, h_t_prev) + bh)
                o_t = Woh.dot(h_t) + bo
                return [o_t, h_t]
        else:
            def forward_prop_step(x_t, h_t_prev):
                Sig = 2*margin*(sigmoid(P)-0.5) + sig_mean
                Hu1SigHv1 = T.set_subtensor(Sig[-1], Sig[-1] * U[-1,-1] * V[-1,-1])
                h_t = f_act(Whi.dot(x_t) + svd_H_wy(U[:,:-1], V[:,:-1], Hu1SigHv1,  h_t_prev) + bh)
                o_t = Woh.dot(h_t) + bo
                return [o_t, h_t]

        [o_scan, _], _ = theano.scan(
            forward_prop_step,
            sequences=[x_scan],
            outputs_info=[None, h0],
            n_steps=seq_len
        )

        if obj == 'c': # classification task
            self.y = T.bscalar('y')
            self.o = f_out(o_scan[-1])
            #obj function to compute grad, use dropout
            self.cost = T.nnet.categorical_crossentropy(self.o, T.eye(n_out)[self.y])
            #compute accuracy use average of dropout rate
            self.accuracy = T.switch(T.eq(T.argmax(self.o), self.y), 1., 0.)
            self.prediction = np.argmax(self.o)
        elif obj == 'r': # regression task
            self.y = T.dscalar('y')
            self.o = o_scan[-1]
            #obj function to compute grad, use dropout
            self.cost = (self.o[0] - self.y)**2
            #compute accuracy use average of dropout rate
            self.accuracy = (self.o[0] - self.y)**2
            self.prediction = self.o[0]

        self.max_singular = 2*margin*(sigmoid(T.max(self.p[2]))-0.5) + sig_mean
        self.min_singular = 2*margin*(sigmoid(T.min(self.p[2]))-0.5) + sig_mean

        self.optimiser = sgd_optimizer(self, 'svdRNN')
