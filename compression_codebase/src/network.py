import tensorflow as tf
import common_config as config #import *
import pretrain_config as preconf
import finetune_config as LRconf
import numpy as np
import pandas as pd
import math
import utils
import pdb

def LSTM(x,weights,biases):
    embed = tf.nn.embedding_lookup(weights['w1'], x)
    if preconf.batch_norm == 'true':
        rnn_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(config.rec_units)
    else:
        rnn_cell = tf.contrib.rnn.LSTMCell(config.rec_units, forget_bias=1.0, use_peepholes=True)
        if preconf.dropout == 'true':
            rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=preconf.DROPOUT_PROB)
    layer_1,state = tf.nn.dynamic_rnn(rnn_cell, embed ,dtype=tf.float32)
    layer_1 = tf.reduce_mean(layer_1,axis=1)
    y = tf.matmul(layer_1, weights['w2']) + biases['b2']
    return y

def LSTM_LR(x,weights,biases):
    embed_0 = tf.nn.embedding_lookup(weights['w11'], x)
    emb_unstack = tf.unstack(embed_0)
    processed = [] # this will be the list of processed tensors
    for t in emb_unstack:
        result_tensor = tf.matmul(t, weights['w12'])
        processed.append(result_tensor)
    embed = tf.concat([processed], 0)
    #pdb.set_trace()
    if LRconf.batch_norm == 'true':
        rnn_cell = tf.contrib.rnn.LayerNormLSTMCell(config.rec_units)
    else:
        rnn_cell = tf.contrib.rnn.LSTMCell(config.rec_units, forget_bias=1.0, use_peepholes=True)
        if LRconf.dropout == 'true':
            rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=LRconf.DROPOUT_PROB)
    layer_1,state = tf.nn.dynamic_rnn(rnn_cell, embed ,dtype=tf.float32)
    layer_1 = tf.reduce_mean(layer_1,axis=1)
    y = tf.matmul(layer_1, weights['w2']) + biases['b2']
    return y
def DAN(x,weights,biases):
    embed = tf.nn.embedding_lookup(weights['w1'], x)
    embed = tf.reduce_mean(embed,axis=1)
    
    #if preconf.dropout=='true':
    #        embed = tf.nn.dropout(embed, preconf.DROPOUT_PROB)

    layer_1 = tf.matmul(embed, weights['w2']) + biases['b2']
    if preconf.batch_norm == 'true':
        layer_1 = tf.contrib.layers.batch_norm(layer_1, center=True,scale=True,is_training=True,fused=True)
        layer_1 = tf.nn.relu(layer_1)
    else:
        layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.matmul(layer_1, weights['w3']) + biases['b3']
    if preconf.batch_norm == 'true':
        layer_2 = tf.contrib.layers.batch_norm(layer_2, center=True,scale=True,is_training=True,fused=True)
        layer_2 = tf.nn.relu(layer_2)
    else:
        layer_2 = tf.nn.relu(layer_2)

   # layer_3 = tf.matmul(layer_2, weights['w4']) + biases['b4']
   # if preconf.batch_norm == 'true':
   #     layer_2 = tf.contrib.layers.batch_norm(layer_3, center=True,scale=True,is_training=True,fused=True)
   #     layer_2 = tf.nn.relu(layer_3)
   # else:
   #     layer_2 = tf.nn.relu(layer_3)

    y = tf.matmul(layer_2, weights['w5']) + biases['b5']
    return y

def DAN_LR(x,weights,biases):
    embed = tf.nn.embedding_lookup(weights['w11'], x)
    embed = tf.reduce_mean(embed,axis=1)


    layer_1 = tf.matmul(embed, weights['w12'])
    if preconf.batch_norm == 'true':
      layer_1 = tf.contrib.layers.batch_norm(layer_1, center=True,scale=True,is_training=True,fused=True)
      layer_1 = tf.nn.relu(layer_1)
    else:
      layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.matmul(layer_1, weights['w2']) + biases['b2']
    if preconf.batch_norm == 'true':
      layer_2 = tf.contrib.layers.batch_norm(layer_2, center=True,scale=True,is_training=True,fused=True)
      layer_2 = tf.nn.relu(layer_2)
    else:
      layer_2 = tf.nn.relu(layer_2) 
      
    layer_3 = tf.matmul(layer_2, weights['w3']) + biases['b3']
    if preconf.batch_norm == 'true':
      layer_3 = tf.contrib.layers.batch_norm(layer_3, center=True,scale=True,is_training=True,fused=True)
      layer_3 = tf.nn.relu(layer_3)
    else:
      layer_3 = tf.nn.relu(layer_3)

    y = tf.matmul(layer_3, weights['w5']) + biases['b5']
  #if preconf.batch_norm == 'true':
  #    layer_3 = tf.contrib.layers.batch_norm(layer_3, center=True,scale=True,is_training=True,fused=True)
  #    layer_3 = tf.nn.relu(layer_3)
  #else:
  #    layer_3 = tf.nn.relu(layer_3)
  #    layer_3 = tf.nn.relu(layer_3)
    return y

def DAN_Q(x,weights,biases):
    embed = tf.nn.embedding_lookup(weights['w1'], x)

    layer_1 = tf.nn.relu(tf.reduce_mean(embed,axis=1))
    #layer_1 = tf.reduce_mean(embed,axis=1)
    #layer_1 = tf.cast(layer_1, tf.float16)

    layer_2 = tf.nn.relu(tf.matmul(layer_1, weights['w2']) + biases['b2'])
    layer_2 = tf.nn.dropout(layer_2, DROPOUT_PROB)

    layer_3 = tf.nn.relu(tf.matmul(layer_2, weights['w3']) + biases['b3'])

    y = tf.matmul(layer_3, weights['w4']) + biases['b4']
    return y


#def DAN_LR(x,weights,biases):
#    embed = tf.nn.embedding_lookup(weights['w11'], x)
#    layer_11 = tf.reduce_mean(embed,axis=1)
#    layer_12 = tf.nn.relu(tf.matmul(layer_11, weights['w12']))

#    layer_2 = tf.nn.relu(tf.matmul(layer_12, weights['w2']) + biases['b2'])
#    layer_2 = tf.nn.dropout(layer_2, LRConf.DROPOUT_PROB)
#
#    layer_3 = tf.nn.relu(tf.matmul(layer_2, weights['w3']) + biases['b3'])
#
#    y = tf.matmul(layer_3, weights['w4']) + biases['b4']
#    return y

def MLP(x, weights, biases):
    layer_1 = tf.matmul(x, weights['w1']) + biases['b1']
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.matmul(layer_1, weights['w2']) + biases['b2']
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.dropout(layer_2 , DROPOUT_PROB)

    y = tf.matmul(layer_2, weights['w3']) + biases['b3']
    return y

def MLP_LR(x, weights, biases):
    layer_11 = tf.matmul(x, weights['w11']) + biases['b11']
    layer_12 = tf.matmul(layer_11, weights['w12']) + biases['b12']
    layer_12 = tf.nn.relu(layer_12)

    layer_2 = tf.matmul(layer_12, weights['w2']) + biases['b2']
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.dropout(layer_2 , DROPOUT_PROB)

    y = tf.matmul(layer_2, weights['w3']) + biases['b3']
    return y

def MLP_Q(x, weights, biases):
    layer_1 = tf.matmul(x, weights['w1']) + biases['b1']
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.matmul(layer_1, weights['w2']) + biases['b2']
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.dropout(layer_2 , DROPOUT_PROB)

    y = tf.matmul(layer_2, weights['w3']) + biases['b3']
    return y

def BiLSTM(x,weights,biases):
    embed = tf.nn.embedding_lookup(weights['w1'], x)
    embed = x = tf.unstack(embed, config.timesteps, 1)
    rnn_fw_cell = tf.contrib.rnn.LSTMCell(config.rec_units, forget_bias=1.0, use_peepholes=True)
    rnn_bw_cell = tf.contrib.rnn.LSTMCell(config.rec_units, forget_bias=1.0, use_peepholes=True)
    if config.dropout == 'true':
            rnn_fw_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_fw_cell, output_keep_prob=config.DROPOUT_PROB)
            rnn_bw_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_bw_cell, output_keep_prob=config.DROPOUT_PROB)

    layer_1, _, _ = tf.nn.bidirectional_dynamic_rnn(rnn_fw_cell,rnn_bw_cell,embed ,dtype=tf.float32)
    layer_1 = tf.reduce_mean(layer_1,axis=1)
    y = tf.matmul(layer_1, weights['w2']) + biases['b2']
    return y


def GRU(x,weights,biases):
    embed = tf.nn.embedding_lookup(weights['w1'], x)

    rnn_cell = tf.contrib.rnn.GRUCell(rec_units)
    layer_1,state = tf.nn.dynamic_rnn(rnn_cell, embed ,dtype=tf.float32)
    layer_1 = tf.reduce_mean(layer_1,axis=1)
    #layer_1 = layer_1[-1]
    layer_1 = utils.apply_nl(layer_1)
    layer_1 = tf.nn.dropout(layer_1, DROPOUT_PROB)

    layer_2 = tf.matmul(layer_1, weights['w2']) + biases['b2']
    layer_2 = utils.apply_nl(layer_2)
    layer_2 = tf.nn.dropout(layer_2, DROPOUT_PROB)

    layer_3 = tf.matmul(layer_2, weights['w3']) + biases['b3']
    layer_3 = utils.apply_nl(layer_3)

    y = tf.matmul(layer_3, weights['w4']) + biases['b4']
    return y #,embed


