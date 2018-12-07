import tensorflow as tf
import os
import re
import pandas as pd
import numpy as np
import math
from collections import defaultdict
from common_config import *
def load_embedding_from_disks_dense(glove_filename, word2index_vocab):
    """
    Read a GloVe txt file. If `with_indexes=True`, we return a tuple of two dictionnaries
    `(word_to_index_dict, index_to_embedding_array)`, otherwise we return only a direct
    `word_to_embedding_dict` dictionnary mapping from a string to a numpy array.
    """

    word_to_index_dict = dict()
    index_to_embedding_array = []

    j = 0
    with open(glove_filename, 'r') as glove_file:
        for (i,line) in enumerate(glove_file):
            if i==0:
                print line
            split = line.split(' ')
            word = split[0]
            value = word2index_vocab.get(word.lower())
            if value != None:
                representation = split[1:]
                representation = np.array([float(val) for val in representation])

                word_to_index_dict[word] = j
                index_to_embedding_array.append(representation)
                j= j+1

    _WORD_NOT_FOUND = [0.0]* len(representation)  # Empty representation for unknown words.

    #_LAST_INDEX = j
    #print _LAST_INDEX
    word_to_index_dict = defaultdict(lambda: _LAST_INDEX, word_to_index_dict)
    word_to_index_dict['_WORD_NOT_FOUND']=j
    index_to_embedding_array = np.array(index_to_embedding_array + [_WORD_NOT_FOUND])

    return word_to_index_dict, index_to_embedding_array

def test_embedding_compression(k,index_to_embedding):
  #data_path = '/home/ec2-user/Compression_exp/datasets/alexa/'
  #train_df, dev_df, test_df = load.extract()
  #prepro = preprocess.PROCESSED_DATA(train_df["sentence"],train_df["label"])
  #word_to_index, index_to_embedding = utils.load_embedding_from_disks_dense(config.EMBED_FILE, prepro.word2index)
  #U,s,VT = np.linalg.svd(index_to_embedding, full_matrices=True)
  #tf_config = tf.ConfigProto(log_device_placement=True)
  #tf_config.gpu_options.allow_growth=True
  #sess = tf.Session(config=tf_config)
  St,Ut,Vt = tf.svd(index_to_embedding)
  #p = 1 - p
  #k = int(p * (index_to_embedding.shape[0]*index_to_embedding.shape[1]/(index_to_embedding.shape[0]+index_to_embedding.shape[1])))
  Sk = np.diag(St[0:k])
  Uk = Ut[:, 0:k]
  Vk = Vt[0:k, :]
  #pdb.set_trace()
  #print('Blah')
  return Uk
def load_embedding_from_disks_sparse(glove_filename, with_indexes=True):
    """
    Read a GloVe txt file. If `with_indexes=True`, we return a tuple of two dictionnaries
    `(word_to_index_dict, index_to_embedding_array)`, otherwise we return only a direct
    `word_to_embedding_dict` dictionnary mapping from a string to a numpy array.
    """
    if with_indexes:
        word_to_index_dict = dict()
        index_to_embedding_array = []
    else:
        word_to_embedding_dict = dict()
    with open(glove_filename, 'r') as glove_file:
        for (i, line) in enumerate(glove_file):
            split = line.split(' ')
            word = split[0]
            representation = split[1:]
            representation = np.array([float(val) for val in representation])
            if with_indexes:
                word_to_index_dict[word] = i
                index_to_embedding_array.append(representation)
            else:
                word_to_embedding_dict[word] = representation
    _WORD_NOT_FOUND = [0.0]* len(representation)  # Empty representation for unknown words.
    if with_indexes:
        _LAST_INDEX = i + 1
        word_to_index_dict = defaultdict(lambda: _LAST_INDEX, word_to_index_dict)
        index_to_embedding_array = np.array(index_to_embedding_array + [_WORD_NOT_FOUND])
        return word_to_index_dict, index_to_embedding_array
    else:
        word_to_embedding_dict = defaultdict(lambda: _WORD_NOT_FOUND)
        return word_to_embedding_dict

def numpy_pad(batches, pad_val):
    lens = np.array([len(i) for i in batches])
    #import pdb; pdb.set_trace()
    mask = np.arange(lens.max()) < lens[:,None]
    out = pad_val*np.ones(mask.shape, dtype='int32')
    out[mask] = np.concatenate(batches)
    return out

def pad_maxlen(batches, pad_val):
    lens = np.array([len(i) for i in batches])
    #import pdb; pdb.set_trace()
    mask = np.arange(maxlen) < lens[:,None]
    #import pdb; pdb.set_trace()
    out = pad_val*np.ones(mask.shape, dtype='int32')
    out[mask] = np.concatenate(batches)
    return out

def reduce_sum_det(x):
    v = tf.reshape(x, [1, -1])
    return tf.reshape(tf.matmul(v, tf.ones_like(v), transpose_b=True), [])

def apply_nl(ip):
    if nl == 'relu':
        return tf.nn.relu(ip)
    elif nl == 'tanh':
        return tf.nn.tanh(ip)

def get_triangular_lr(iteration, stepsize, base_lr, max_lr):
    """Given the inputs, calculates the lr that should be applicable for this iteration"""
    """http://teleported.in/posts/cyclic-learning-rate/"""
    bump = float(max_lr - base_lr)/float(stepsize)
    cycle = iteration%(2*stepsize)
    if cycle < stepsize:
        lr = base_lr + cycle*bump
    else:
        lr = max_lr - (cycle-stepsize)*bump
    return lr
