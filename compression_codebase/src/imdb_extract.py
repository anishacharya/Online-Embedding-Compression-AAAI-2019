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
This Code Snippet Processes Raw Large IMDB Data and loads into pandas dataframe
DATASET : http://ai.stanford.edu/~amaas/data/sentiment/
PS: This does not pre-process/clean raw text You should run preprocess.py after you load using this script

Author: Anish Acharya <achanish@amazon.com>
'''
def load_directory_data(directory):
    data = {}
    data["sentence"] = []
    data["sentiment"] = []
    for file_path in os.listdir(directory):
        with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
            data["sentence"].append(f.read())
            data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
    return pd.DataFrame.from_dict(data)

# Merge positive and negative examples, add a label column and shuffle.
def load_dataset(directory):
    pos_df = load_directory_data(os.path.join(directory, "pos"))
    neg_df = load_directory_data(os.path.join(directory, "neg"))
    pos_df["label"] = 1
    neg_df["label"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)

# Download and process the dataset files.
def download_and_load_datasets(dataset):
    train_df = load_dataset(os.path.join(dataset,"aclImdb","train")) #(os.path.join(os.path.dirname(dataset), "aclImdb", "train"))
    dev_df = load_dataset(os.path.join(dataset, "aclImdb", "dev"))
    test_df = load_dataset(os.path.join(dataset, "aclImdb", "test"))
    return train_df,dev_df,test_df

def extract():
    tf.logging.set_verbosity(tf.logging.ERROR)
    train_df, dev_df, test_df = download_and_load_datasets(data_path)

    return train_df, dev_df, test_df
