#!/usr/bin/env python
# -*- coding: utf-8 -*-

import custom_config

def print_setting():
    print "Running network: ", custom_config.netType
    print "Input dimension: ", custom_config.n_in
    print "Output dimension: ", custom_config.n_out
    print "Hidden layers: ", custom_config.n_layers
    print "Hidden dimension: ", custom_config.n_h
    print "Number of reflection vectors: ", custom_config.n_r
    print "Singualr margin: ", custom_config.m
    print "Hidden-to-hidden activation function: ", custom_config.activation_Hidden #"leacky relu"
    print "Output activation function: ", custom_config.activation_ooutput #"softmax"
    print "Instance per batch: ", custom_config.batchsize
    print "Validation interval: ", custom_config.validation_int
    print "Number of epoch: ", custom_config.num_epoch
    print "Learning rate: ", custom_config.learning_rate
    print "Droput rate: ", custom_config.dropout
    
    return 

