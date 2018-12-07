from common_config import *
import pandas as pd
import tensorflow as tf
'''
This Code Snippet Reads .pkl File and Loads into df for DataSet Specified in config.py
In Order to generate .pkl file from Raw DataSet : https://github.com/anishacharya/My_Deep_Learning_Library/tree/master/DataSets
./preprocess script already Cleans the Raw File before creating the .pkl file

Supported NLP DataSets : MR, SST2, TREC
For Processing Large IMDB -> imdb_extract.py
For Processing Alexa Data -> alexa_extract.py

Author : Anish Acharya <achanish@amazon.com>
'''
def load_dataset(split,pkl_file):
    #df = pd.DataFrame(corpus.loc(corpus['split']==split))
    corpus = pd.read_pickle(pkl_file)
    return corpus[corpus['split']==split]

def extract():
    data = EXP+'.pkl'
    pkl_file=data_path+'/'+data
    tf.logging.set_verbosity(tf.logging.ERROR)
    train_df = load_dataset('train',pkl_file)
    train_df = train_df.sample(frac=1)
    dev_df = load_dataset('dev',pkl_file)
    test_df = load_dataset('test',pkl_file)
    return train_df, dev_df,test_df

#def create_vocab():
#    data = EXP+'.pkl'
#    pkl_file=data_path+'/'+data
#    train_df = load_dataset('train',pkl_file)
