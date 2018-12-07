import tensorflow as tf
import os
import re
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
import math
import common_config as config

'''
author: Anish Acharya <achanish@amazon.com>
'''
#vocab = '/home/ec2-user/EMNLP_2018/datasets/aclImdb/imdb.vocab'

class PROCESSED_DATA:
    def __init__(self, reviews,labels):
        np.random.seed(1)
        self.pre_process_data(reviews, labels )
        self.VOCAB_SIZE = len(self.review_vocab)
        self.NUM_CLASSES = self.label_vocab_size

    def pre_process_data(self, reviews, labels):
        review_vocab = set()
        if config.EXP == 'IMDB':
            self.review_vocab = open(vocab).read().split("\n")
        else:
            for review in reviews:
                text = review.lower()
                text = re.sub(r"\. \. \.", "\.", text)
                text = re.sub(r"[^A-Za-z0-9(),!?\'\`\.]", " ", text)
                # text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
                text = re.sub(r"\'s", " \'s", text)
                text = re.sub(r"\'ve", " \'ve", text)
                text = re.sub(r"n\'t", " n\'t", text)
                text = re.sub(r"\'re", " \'re", text)
                text = re.sub(r"\'d", " \'d", text)
                text = re.sub(r"\'ll", " \'ll", text)
                text = re.sub(r",", " , ", text)
                text = re.sub(r"!", " ! ", text)
                text = re.sub(r"\(", " ( ", text)
                text = re.sub(r"\)", " ) ", text)
                text = re.sub(r"\?", " ? ", text)
                text = re.sub(r"\s{2,}", " ", text)
                #text = re.sub(r"<br />", " ", text)
                #text = re.sub(r"[^a-z]", " ", text)
                #text = text.split(" ")
                #stops = set(stopwords.words("english"))
                #text = [w for w in text if not w in stops]
                for word in text.split(" "):
                    review_vocab.add(word)

            self.review_vocab = list(review_vocab)

        label_vocab = set()
        for label in labels:
            label_vocab.add(label)

        self.label_vocab = list(label_vocab)
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)

        self.word2index = {}
        for i, word in enumerate(self.review_vocab):
            self.word2index[word.lower()] = i

        self.label2index = {}
        for i, label in enumerate(self.label_vocab):
            self.label2index[label] = i
    ### Generate Batches
    def get_batch_sparse(self, df_data, df_target, i,batch_size):
        batches = []
        results = []
        texts = df_data[i*batch_size:i*batch_size+batch_size]
        categories = df_target[i*batch_size:i*batch_size+batch_size]

        for text in texts:
            layer = np.zeros(self.VOCAB_SIZE,dtype=float)
            for word in text.split(' '):
                value = self.word2index.get(word.lower())
                if value != None:
                    layer[self.word2index[word.lower()]] += 1
            batches.append(layer)

        for category in categories:
            y = np.zeros((self.NUM_CLASSES),dtype=float)
            if self.label2index.get(category) != None:
                 y[self.label2index[category]] = 1
            results.append(y)

        return np.array(batches),np.array(results)

    def get_batch_dense(self, df_data, df_target, i, batch_size, word_to_index):
        batches = []
        results = []

        texts = df_data[i*batch_size:i*batch_size+batch_size]
        categories = df_target[i*batch_size:i*batch_size+batch_size]

        for text,category in zip(texts,categories):
            layer = []
            if len(text.split(' ')) > config.minlen:
                i=0
                for word in text.split(' '):
                    if i < config.maxlen:
                        value = word_to_index.get(word.lower())
                        if value != None:
                            layer.append(value)
                    i+=1
                batches.append(np.asarray(layer))
                y = np.zeros((self.NUM_CLASSES +1),dtype=float)
                if self.label2index.get(category) != None:
                    y[self.label2index[category]] = 1
                else:
                    y[self.NUM_CLASSES] = 1  # handle unseen intent
                results.append(y)
        return np.array(batches),np.array(results)

    def get_dense(self, df_data, df_target, word_to_index):
        batches = []
        results = []

        texts = df_data#[i*batch_size:i*batch_size+batch_size]
        categories = df_target#[i*batch_size:i*batch_size+batch_size]

        for text,category in zip(texts,categories):
            layer = [] #np.zeros(self.VOCAB_SIZE,dtype=float)
            if len(text.split(' ')) > config.minlen:
                j = 0
                for word in text.split(' '):
                    if j < config.maxlen:
                    #value = self.word2index.get(word.lower())
                        value = word_to_index.get(word.lower())
                        if value != None:
                            layer.append(value)
                        #layer[self.word2index[word.lower()]] += 1
                        j +=1
                batches.append(np.asarray(layer))
            #for category in categories:

                y = np.zeros((self.NUM_CLASSES +1),dtype=float)
                if self.label2index.get(category) != None:
                    y[self.label2index[category]] = 1
                else:
                    y[self.NUM_CLASSES] = 1  # handle unseen intent
                results.append(y)

#            else:
#                print('Dropped Sentence Length: ',len(text.split(' ')))

        return np.array(batches),np.array(results)
