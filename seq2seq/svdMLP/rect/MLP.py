import numpy as np
import theano as theano
import theano.tensor as T
from sgd import sgd_optimizer
from Layers import HiddenLayer

def leaky_relu(z):
    return T.switch(z < 0, .1 * z, z)


def linear(z):
    return z

def sigmoid(z):
    return T.nnet.nnet.sigmoid(z)

def softmax(z):
    return T.nnet.softmax(z)[0]

def hyperbolic_tangent(z):
    return T.tanh(z)


class MLP():
    def __init__(self, rng, n_in,  n_out, n_h, f_act=leaky_relu, obj='single', dropout_rate =0):
        '''
        :param rng: Numpy RandomState
        :param n_in: Input dimension (int)
        :param n_out: Output dimension (int)
        :param n_h: Hidden dimension (list of int)
        :param f_act: Hidden-to-hidden activation function
        :param f_out: Output activation function
        '''
        if obj=='single':
            f_out = softmax
        elif obj=='multi':
            f_out = sigmoid
        self.x = T.vector()

        # construct hidden layers
        n_layers = len(n_h)
        first_hiddenLayer = HiddenLayer(
            rng=rng,
            input=self.x,
            n_in=n_in,
            n_out=n_h[0],
            activation=f_act,
            dropout_rate = dropout_rate,
            nametag='0'
        )

        self.hidden_layers = [first_hiddenLayer]
        self.p = first_hiddenLayer.params[:]

        for i in range(n_layers-1):
            cur_hiddenLayer = HiddenLayer(
                rng=rng,
                input=self.hidden_layers[-1].output,
                n_in=n_h[i],
                n_out=n_h[i+1],
                activation=f_act,
                dropout_rate = dropout_rate,
                nametag=str(i+1)
                )
            self.hidden_layers.append(cur_hiddenLayer)
            self.p.extend(cur_hiddenLayer.params[:])

        # params for output layer

        self.outputLayer = HiddenLayer(
            rng=rng,
            input=self.hidden_layers[-1].output,
            n_in=n_h[-1],
            n_out=n_out,
            activation=f_out,
            dropout_rate = 0,
            nametag='o'
        )
        self.p.extend(self.outputLayer.params[:])

        self.n_layers = n_layers + 1
        self.obj = obj
        if obj=='single':
            self.y = T.bscalar('y')
            self.o = self.outputLayer.output
            self.o_predict = self.outputLayer.predict_output
            self.cost = T.nnet.categorical_crossentropy(self.o, T.eye(n_out)[self.y])
            self.accuracy = T.switch(T.eq(T.argmax(self.o_predict), self.y), 1., 0.)
            self.prediction = np.argmax(self.o_predict)
        elif obj=='multi':
            self.y = T.bvector('y')
            self.o = self.outputLayer.output
            self.o_predict = self.outputLayer.predict_output
            self.cost = T.nnet.binary_crossentropy(self.o, self.y).mean()
            self.prediction = T.argsort(self.o_predict)
            self.accuracy = self.y[T.argmax(self.o_predict)]
            self.accuracy3 = (1.0/3.0) * (self.y[self.prediction[-3]]+self.y[self.prediction[-2]]+self.y[self.prediction[-1]])
            self.accuracy5 = (1.0/5.0) * (self.y[self.prediction[-5]]+self.y[self.prediction[-4]]+self.y[self.prediction[-3]]+self.y[self.prediction[-2]]+self.y[self.prediction[-1]])

        self.optimiser = sgd_optimizer(self, 'MLP')
