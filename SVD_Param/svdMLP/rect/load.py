#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
import cPickle as pickle

datasets_dir = os.getcwd() + '/../../data/'

def one_hot(x,n):
	if type(x) == list:
		x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x),n))
	o_h[np.arange(len(x)),x] = 1
	return o_h

def mnist(ntrain=60000,ntest=10000,onehot=True):
	data_dir = os.path.join(datasets_dir,'mnist/')
	fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trX = loaded[16:].reshape((60000,28*28)).astype(float)

	fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trY = loaded[8:].reshape((60000))

	fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teX = loaded[16:].reshape((10000,28*28)).astype(float)

	fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teY = loaded[8:].reshape((10000))

	trX = trX/255.
	teX = teX/255.

	trX = trX[:ntrain]
	trY = trY[:ntrain]

	teX = teX[:ntest]
	teY = teY[:ntest]

	if onehot:
		trY = one_hot(trY, 10)
		teY = one_hot(teY, 10)
	else:
		trY = np.asarray(trY)
		teY = np.asarray(teY)

	return trX,teX,trY,teY
#----------------------------------------------
UCR50words_dir = datasets_dir + 'UCR_TS_Archive_2015/50words/'
def UCR50words(onehot=False):
    training_data = np.loadtxt(UCR50words_dir+'50words_TRAIN', delimiter=',')
    test_data = np.loadtxt(UCR50words_dir+'50words_TEST', delimiter=',')

    trX = training_data[:,1:].astype(float)
    trY = training_data[:,0].astype(int)-1
    teX = test_data[:,1:].astype(float)
    teY = test_data[:,0].astype(int)-1

    if onehot:
		trY = one_hot(trY, 50)
		teY = one_hot(teY, 50)
    else:
		trY = np.asarray(trY)
		teY = np.asarray(teY)

    return trX,teX,trY,teY

#----------------------------------------------
def UCR(name,onehot=False):
    cur_dataset_dir = datasets_dir + 'UCR_TS_Archive_2015/'+name
    training_data = np.loadtxt(cur_dataset_dir+'/'+name+'_TRAIN', delimiter=',')
    test_data = np.loadtxt(cur_dataset_dir+'/'+name+'_TEST', delimiter=',')

    trX = training_data[:,1:].astype(float)
    trY = training_data[:,0].astype(int)
    teX = test_data[:,1:].astype(float)
    teY = test_data[:,0].astype(int)

    n_class = max(np.max(trY),np.max(teY))

    if onehot:
		trY = one_hot(trY, n_class)
		teY = one_hot(teY, n_class)
    else:
		trY = np.asarray(trY)
		teY = np.asarray(teY)

    return trX,teX,trY,teY

#-----------------------------------------------
cifar10=datasets_dir+'/cifar-10-batches-py'

def load_CIFAR_batch(filename):
    ''' load single batch of cifar '''
    with open(filename, 'r') as f:
        datadict = pickle.load(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def CIFAR10(ROOT=cifar10, onehot=True):
    ''' load all of cifar '''
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))

    if onehot:
        Ytr = one_hot(Ytr, 10)
        Yte = one_hot(Yte, 10)
    else:
        Ytr = np.asarray(Ytr)
        Yte = np.asarray(Yte)

    Xtr = Xtr/255.
    Xte = Xte/255.

    return Xtr, Xte, Ytr, Yte

#----------------------------------------------------
def Adding_task(filename=datasets_dir+'/Adding_task/data', ntrain=50000, ntest=1000):
    data = np.loadtxt(filename, delimiter=',').astype(float)
    X = data[:,1:]; Y = data[:,0]
    assert(ntrain+ntest <= X.shape[0])
    return X[0 : ntrain], X[ntrain : ntrain + ntest], Y[0 : ntrain], Y[ntrain : ntrain + ntest]

