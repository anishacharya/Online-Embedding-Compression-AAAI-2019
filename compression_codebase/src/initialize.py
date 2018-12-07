import sys
import tensorflow as tf
from common_config import *
import pretrain_config as preconf

def initialize_nw(net_Type, input_dim, output_dim,embed_dim= EMBEDDING_DIM):
    if net_Type =='DAN' :
        weights = {
            'w1': tf.Variable(tf.constant(0.0, shape=[input_dim, embed_dim]),trainable=True, name="W1"),
            'w2': tf.Variable(tf.random_normal([embed_dim, hidden1],
                stddev=preconf.std),name = 'W2'),
            'w3': tf.Variable(tf.random_normal([hidden1, hidden2],
                stddev=preconf.std),name = 'W3'),
            #'w4': tf.Variable(tf.random_normal([hidden2, hidden3],
                #stddev=preconf.std),name = 'W4'),

            'w5': tf.Variable(tf.random_normal([hidden2, output_dim],
                stddev=preconf.std),name = 'W5')

        }

        biases = {
            'b2': tf.Variable(tf.random_normal([hidden1]),name = 'b2'),
            'b3': tf.Variable(tf.random_normal([hidden2]),name = 'b3'),
            #'b4': tf.Variable(tf.random_normal([hidden3]),name = 'b4'),
            'b5': tf.Variable(tf.random_normal([output_dim]),name = 'b5'),

        }
    elif netType =='LSTM' or 'BiLSTM' or 'GRU':
        weights = {
            'w1': tf.Variable(tf.constant(0.0, shape=[input_dim, embed_dim]),trainable=True, name="W1"),
            'w2': tf.Variable(tf.random_normal([rec_units, output_dim], mean=preconf.mean, stddev=preconf.std),name = 'W2')
        }
        biases = {
            'b2': tf.Variable(tf.random_normal([output_dim], stddev=preconf.std),name = 'b2')
        }
    elif netType =='LSTM_LR' or 'BiLSTM_LR' or 'GRU_LR':
        ''' Creates a Low Rank Model From scratch no SVD initialization '''
        p = 1 - Percentage_Reduction
        k = int(p * (input_dim*embed_dim)/(input_dim + embed_dim))

        weights = {
            'w11': tf.Variable(tf.constant(0.0, shape=[input_dim, k] ),trainable=True, name="W1"),
            'w12': tf.Variable(tf.random_normal([k, embed_dim]), name = 'W12'),
            'w2': tf.Variable(tf.random_normal([embed_dim, output_dim], stddev=std),name = 'W2')
        }
        biases = {
            'b12': tf.Variable(tf.random_normal([embed_dim]),name = 'b12'),
            'b2': tf.Variable(tf.random_normal([output_dim], stddev=std),name = 'b2')
        }
    elif netType == 'DAN_LR':
        ''' Creates a Low Rank Model From scratch no SVD initialization '''
        p = 1 - Percentage_Reduction
        k = int(p * (input_dim*embed_dim)/(input_dim + embed_dim))
        weights = {
            'w11': tf.Variable(tf.constant(0.0, shape=[input_dim, k]),trainable=True, name="W11"),
            'w12': tf.Variable(tf.random_normal([k, embed_dim]), name = 'W12'),

            'w2': tf.Variable(tf.random_normal([embed_dim, hidden1]),name = 'W2'),
            'w3': tf.Variable(tf.random_normal([hidden1, hidden2]),name = 'W3'),
            'w5': tf.Variable(tf.random_normal([hidden2, output_dim]),name = 'W5')
        }

        biases = {
            'b2': tf.Variable(tf.random_normal([hidden1]),name = 'b2'),
            'b3': tf.Variable(tf.random_normal([hidden2]),name = 'b3'),
            'b5': tf.Variable(tf.random_normal([output_dim]),name = 'b5')
        }

    elif net_Type =='MLP':
        weights = {
            'w1': tf.Variable(tf.random_normal([input_dim,hidden1]), name="W1"),
            'w2': tf.Variable(tf.random_normal([hidden1, hidden2]),name = 'W2'),
            'w3': tf.Variable(tf.random_normal([hidden2, output_dim]),name = 'W3'),
        }

        biases = {
            'b1': tf.Variable(tf.random_normal([hidden1]),name = 'b1'),
            'b2': tf.Variable(tf.random_normal([hidden2]),name = 'b2'),
            'b3': tf.Variable(tf.random_normal([output_dim]),name = 'b3'),
        }
    elif netType == 'MLP_LR':
        ''' Creates a Low Rank Model From scratch no SVD initialization '''
        p = 1 - Percentage_Reduction
        k = int(p * (input_dim*hidden1)/(input_dim + hidden1))
        weights = {
            'w11': tf.Variable(tf.random_normal([input_dim, k]), name = 'W11'),
            'w12': tf.Variable(tf.random_normal([k, hidden1]), name = 'W12'),
            'w2': tf.Variable(tf.random_normal([hidden1, hidden2]),name = 'W2'),
            'w3': tf.Variable(tf.random_normal([hidden2, output_dim]),name = 'W3')
        }

        biases = {
            'b11': tf.Variable(tf.random_normal([k]),name = 'b11'),
            'b12': tf.Variable(tf.random_normal([hidden1]),name = 'b12'),
            'b2': tf.Variable(tf.random_normal([hidden2]),name = 'b2'),
            'b3': tf.Variable(tf.random_normal([output_dim]),name = 'b3')
        }
    return weights,biases
