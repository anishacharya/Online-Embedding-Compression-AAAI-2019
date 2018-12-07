import os
'''
author: Anish Acharya <achanish@amazon.com>
'''
EMBED_FILE = '/home/ec2-user/DataSets/glove.6B.300d.txt'
EMBEDDING_DIM = 300

Emb_Compress = 'false' ## True If we want Offline Compression 
compression = 0.1     

minlen = 0
maxlen = 40
timesteps = maxlen

#Network Hyperparameters
#LSTM
rec_units = 150 #EMBEDDING_DIM #256
#DAN
hidden1 = 30
hidden2 = 25
hidden3 = 256
######## ------ NETWORK CREATION CONFIGS ---------- ####
EXP = 'SST2'                              # Options: 'IMDB', 'alexa','MR','SST2','TREC'
domain = 'Books'                           # alexa domain : Options: Books Books_large
netType = 'LSTM'                             # Options: 'MLP', 'DAN', 'LSTM'


vocab = '/home/ec2-user/DataSets/aclImdb/imdb.vocab'
if EXP =='alexa':
    model_path = '/home/ec2-user/Compression_exp/models/'
    data_path = '/home/ec2-user/DataSets/alexa/'
    data_path = data_path + domain
else:
    model_path = '/home/ec2-user/Compression_exp/models/'
    data_path = '/home/ec2-user/DataSets/'
