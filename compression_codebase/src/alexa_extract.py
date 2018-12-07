#Imports
import tensorflow as tf
import os
import re
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
import math
from common_config import *
'''
This Code Snippet Reads Alexa Training Data Format and Puts them into pandas DataFrame
The Train, Test, Dev Files have to be in respective folders
Author: Rahul Goel <goerahul@amazon.com>, Anish Acharya <achanish@amazon.com>
'''
def load_directory_data(directory):
    data = {}
    data["sentence"] = []
    data["label"] = []
    for file_path in os.listdir(directory):
        with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
            for line in f:
                domain, intent, annotation_str = line.split("\t")[0:3]
                parts = annotation_str.split(" ")
                words = []
                labels= []
                for part in parts:
                    word, label = part.split("|")
                    words.append(word)
                    labels.append(label)
                data["sentence"].append(" ".join(words))
                data["label"].append(intent)
    return pd.DataFrame.from_dict(data)

# Merge positive and negative examples, add a label column and shuffle.
def load_dataset(directory):
    pos_df = load_directory_data(os.path.join(directory))
    return pos_df

# Download and process the dataset files.
path_to_data= data_path #'/home/ec2-user/EMNLP_2018/datasets/alexa'
def download_and_load_datasets(force_download=False):
    train_df = load_dataset(os.path.join(path_to_data, "train"))
    dev_df = load_dataset(os.path.join(path_to_data, "dev"))
    test_df = load_dataset(os.path.join(path_to_data, "test"))
    return train_df, dev_df, test_df

def extract():
    tf.logging.set_verbosity(tf.logging.ERROR)
    train_df, dev_df, test_df = download_and_load_datasets(data_path)
    train_df = train_df.sample(frac=1)
    return train_df, dev_df, test_df
