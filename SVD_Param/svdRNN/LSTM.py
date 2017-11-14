import numpy as np
import theano as theano
import theano.tensor as T
from tensor_ops import H_wy
from sgd import sgd_optimizer
from activations import *
import time

class LSTM():
    def __init__(self, rng, n_in, n_out, n_h, dropout=0, sigma_g= sigmoid, sigma_c= hyperbolic_tangent, sigma_h = hyperbolic_tangent, sigma_y = softmax,dropout_rate=0, obj='c'):
        '''
        :param rng: Numpy RandomState
        :param n_in: Input dimension (int)
        :param n_out: Output dimension (int)
        :param n_h: Hidden dimension (int)
        :param sigma_g, sigma_c, sigma_h, sigma_y: activation functions
        :param dropout_rate: dropout rate (float)
        :param obj: objective type, 'c' for classification with cross entropy loss, 'r' for regression with MSE loss. (['c','r'])
        '''

        Wf_ = rng.uniform(-np.sqrt(6. / (n_in + n_h)), np.sqrt(6. / (n_in + n_h )), (n_h, n_in))
        Uf_ = rng.uniform(-np.sqrt(6. / (n_h + n_h)), np.sqrt(6. / (n_h + n_h )), (n_h, n_h))
        bf_ = np.zeros( n_h)

        Wi_ = rng.uniform(-np.sqrt(6. / (n_in + n_h)), np.sqrt(6. / (n_in + n_h )), (n_h, n_in))
        Ui_ = rng.uniform(-np.sqrt(6. / (n_h + n_h)), np.sqrt(6. / (n_h + n_h )), (n_h, n_h))
        bi_ = np.zeros( n_h)

        Wo_ = rng.uniform(-np.sqrt(6. / (n_in + n_h)), np.sqrt(6. / (n_in + n_h )), (n_h, n_in))
        Uo_ = rng.uniform(-np.sqrt(6. / (n_h + n_h)), np.sqrt(6. / (n_h + n_h )), (n_h, n_h))
        bo_ = np.zeros( n_h)

        Wc_ = rng.uniform(-np.sqrt(6. / (n_in + n_h)), np.sqrt(6. / (n_in + n_h )), (n_h, n_in))
        Uc_ = rng.uniform(-np.sqrt(6. / (n_h + n_h)), np.sqrt(6. / (n_h + n_h )), (n_h, n_h))
        bc_ = np.zeros( n_h)

        Wy_ = rng.uniform(-np.sqrt(6. / (n_out + n_h)), np.sqrt(6. / (n_out + n_h )), (n_out, n_h))
        by_ = np.zeros( n_out)

        h0_ = rng.uniform(-np.sqrt(3. / (2. * n_h)), np.sqrt(3. / (2. * n_h)), n_h)
        c0_ = rng.uniform(-np.sqrt(3. / (2. * n_h)), np.sqrt(3. / (2. * n_h)), n_h)

        # Theano: Created shared variables
        Wf = theano.shared(name='Wf', value = Wf_.astype(theano.config.floatX))
        Uf = theano.shared(name='Uf', value = Uf_.astype(theano.config.floatX))
        bf = theano.shared(name='bf', value = bf_.astype(theano.config.floatX))

        Wi = theano.shared(name='Wi', value = Wi_.astype(theano.config.floatX))
        Ui = theano.shared(name='Ui', value = Ui_.astype(theano.config.floatX))
        bi = theano.shared(name='bi', value = bi_.astype(theano.config.floatX))

        Wo = theano.shared(name='Wo', value = Wo_.astype(theano.config.floatX))
        Uo = theano.shared(name='Uo', value = Uo_.astype(theano.config.floatX))
        bo = theano.shared(name='bo', value = bo_.astype(theano.config.floatX))

        Wc = theano.shared(name='Wc', value = Wc_.astype(theano.config.floatX))
        Uc = theano.shared(name='Uc', value = Uc_.astype(theano.config.floatX))
        bc = theano.shared(name='bc', value = bc_.astype(theano.config.floatX))

        Wy = theano.shared(name='Wy', value = Wy_.astype(theano.config.floatX))
        by = theano.shared(name='by', value = by_.astype(theano.config.floatX))

        h0 = theano.shared(name='h0', value = h0_.astype(theano.config.floatX))
        c0 = theano.shared(name='c0', value = c0_.astype(theano.config.floatX))

        self.p = [Wf, Uf, bf, Wi, Ui, bi, Wo, Uo, bo, Wc, Uc, bc, Wy, by, c0, h0]


        seq_len = T.iscalar('seq_len')
        self.seq_len = seq_len

        self.x = T.vector()
        x_scan = T.reshape(self.x, [seq_len, n_in], ndim = 2)


        if dropout_rate > 0:
            np.random.seed(int(time.time()))
            # for training
            def masked_forward_prop_step(x_t, h_t_prev, c_t_prev):
                f_t = sigma_g( Wf.dot(x_t) + Uf.dot(h_t_prev) + bf)
                i_t = sigma_g( Wi.dot(x_t) + Ui.dot(h_t_prev) + bi)
                o_t = sigma_g( Wo.dot(x_t) + Uo.dot(h_t_prev) + bo)
                c_t = i_t * sigma_c( Wc.dot(x_t) + Uc.dot(h_t_prev) + bc)
                c_t += c_t_prev * f_t
                h_t = o_t * sigma_h(c_t)
                y_t = Wy.dot(h_t) + by
                mask = np.random.binomial(np.ones(n_h, dtype=int),1.0-dropout_rate)
                masked_h_t = h_t * T.cast(mask, theano.config.floatX)

                return [y_t, masked_h_t, c_t]

            # for testing
            def forward_prop_step(x_t, h_t_prev, c_t_prev):
                f_t = sigma_g( Wf.dot(x_t) + Uf.dot(h_t_prev) + bf)
                i_t = sigma_g( Wi.dot(x_t) + Ui.dot(h_t_prev) + bi)
                o_t = sigma_g( Wo.dot(x_t) + Uo.dot(h_t_prev) + bo)
                c_t = i_t * sigma_c( Wc.dot(x_t) + Uc.dot(h_t_prev) + bc)
                c_t += c_t_prev * f_t
                h_t = o_t * sigma_h(c_t)
                h_t = (1.0-dropout_rate)*h_t
                y_t = Wy.dot(h_t) + by

                return [y_t, h_t, c_t]


            [o_train, _, _], _ = theano.scan(
                masked_forward_prop_step,
                sequences=[x_scan],
                outputs_info=[None, h0, c0],
                n_steps=seq_len
            )

            [o_test, _, _], _ = theano.scan(
                forward_prop_step,
                sequences=[x_scan],
                outputs_info=[None, h0, c0],
                n_steps=seq_len
            )

        else:
            def forward_prop_step(x_t, h_t_prev, c_t_prev):
                f_t = sigma_g( Wf.dot(x_t) + Uf.dot(h_t_prev) + bf)
                i_t = sigma_g( Wi.dot(x_t) + Ui.dot(h_t_prev) + bi)
                o_t = sigma_g( Wo.dot(x_t) + Uo.dot(h_t_prev) + bo)
                c_t = i_t * sigma_c( Wc.dot(x_t) + Uc.dot(h_t_prev) + bc)
                c_t += c_t_prev * f_t
                h_t = o_t * sigma_h(c_t)
                y_t = Wy.dot(h_t) + by

                return [y_t, h_t, c_t]


            [o_train, _, _], _ = theano.scan(
                forward_prop_step,
                sequences=[x_scan],
                outputs_info=[None, h0, c0],
                n_steps=seq_len
            )
            o_test = o_train


        if obj == 'c': # classification task
            self.y = T.bscalar('y')
            self.o_train = sigma_y(o_train[-1])
            self.o_test = sigma_y(o_test[-1])
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

        self.optimiser = sgd_optimizer(self, 'LSTM')

