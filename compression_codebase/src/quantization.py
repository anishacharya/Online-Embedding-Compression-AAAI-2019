rom config import *
from imdb_extract import *
from imdb_preprocess import *
import network as nw
from initialize import *
import alexa_extract as alexa
from utils import *

import time
import pandas as pd
import numpy as np

### ------  Pre-Process and load Data ---- ###
if EXP =='alexa':
    print("Preparing Alexa Data")
    train_df, test_df = alexa.extract()
elif EXP =='IMDB':
    print("Preparing Large MR Data")
    train_df, test_df = extract()
imdb = IMDB(train_df["sentence"],train_df["polarity"])


## -------------- TF Graph  --------------- ###
## Define and Initialize Network
tf.reset_default_graph()

weights, biases = initialize_nw(netType, imdb.VOCAB_SIZE, imdb.NUM_CLASSES+1)
x = tf.placeholder(tf.float32, [None, imdb.VOCAB_SIZE])
y_ = tf.placeholder(tf.float32, [None, imdb.NUM_CLASSES+1])

elif netType == 'DAN' or netType == 'LSTM':
    print('Creating Vocab From Glove')
    word_to_index, index_to_embedding = load_embedding_from_disks_dense(EMBED_FILE, imdb.word2index)
    vocab_size = index_to_embedding.shape[0]
    weights, biases = initialize_nw(netType, vocab_size, imdb.NUM_CLASSES+1)
    embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, EMBEDDING_DIM])
    y_ = tf.placeholder(tf.float32, [None, imdb.NUM_CLASSES + 1])
if netType == 'DAN':
    y = nw.DAN(x,weights,biases)
if netType == 'LSTM':
    y = nw.LSTM(x,weights,biases)
### ----- Define Forward Prop ------- ###

## Evaluation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar('accuracy', accuracy)
### ----- Session Configs etc --------- ###
## Saving and Starting Session
saver = tf.train.Saver()
init = tf.global_variables_initializer()
## GPU Configurations
config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth=True
## Session Start
sess = tf.Session(config=config)
sess.run(init)
if netType == 'DAN':
    embedding_init = weights['w1'].assign(embedding_placeholder)
    sess.run(embedding_init,feed_dict={embedding_placeholder: index_to_embedding})

## Summary Writer
merged = tf.summary.merge_all()
saver.restore(sess, pretrained_model_path)

sample_size=len(test_df["sentence"])
total_batch = int(sample_size / BATCH_SIZE )

avg_acc_pre = 0
avg_acc= 0
for i in range(total_batch):
    if netType == 'DAN' or netType =='DAN_LR':
        try:
            batch_xt, batch_yt = imdb.get_batch_dense(test_df["sentence"], test_df["polarity"], i, BATCH_SIZE, word_to_index)
            batch_xt = numpy_pad(batch_xt, pad_val)
            c,summary = sess.run([accuracy,merged], feed_dict={x: batch_xt, y_: batch_yt})
            avg_acc_pre += c
        except:
            pass
    else:
        try:
            batch_xt, batch_yt = imdb.get_batch_sparse(test_df["sentence"], test_df["polarity"], i, BATCH_SIZE)
            c,summary = sess.run([accuracy,merged], feed_dict={x: batch_xt, y_: batch_yt})
            avg_acc_pre += c
            print('computing')
        except:
            pass
    print (avg_acc_pre/total_batch)
    #train_writer.add_summary(summary, epoch*i + i)
### ----- Quantization ------------- ###
### Get the Weights and Biases into Numpy Array for LP Casting
if quantization_bits == 16:
    bit = tf.float16
else:
    bit = tf.float8

if netType == 'MLP':
    var_np = []
    for variable in tf.trainable_variables():
        print variable
        var_np.append(sess.run(variable))
    print('Resetting Graph')
    tf.reset_default_graph()
    weights = {}
    biases = {}
    print(len(var_np))
    j=1
    i=0
    mid = len(var_np)/2
    while i < mid:
        weights['w'+str(j)] = tf.Variable(tf.cast(var_np[i], dtype=bit), name='W'+str(j))
        biases['b'+str(j)] = tf.Variable(tf.cast(var_np[i+mid], dtype=bit), name='b'+str(j))
        print('Casting Tensors to Lower Precision')
        print weights['w'+ str(j) ]
        print biases['b'+ str(j)]
        j += 1
        i += 1
    print weights
    ### ---- Forward Prop ---- ###
    x = tf.placeholder(bit, [None, imdb.VOCAB_SIZE])
    y_ = tf.placeholder(bit, [None, imdb.NUM_CLASSES])
    y = MLP_Q(x, weights, biases)

## Evaluation
saver = tf.train.Saver()
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)
init = tf.global_variables_initializer()
## GPU Configurations
config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.run(init)
merged = tf.summary.merge_all()

start_time=time.time()
avg_acc = 0
for i in range(total_batch):
    batch_xt, batch_yt = imdb.get_batch_sparse(test_df["sentence"], test_df["polarity"], i, BATCH_SIZE)
    c,summary = sess.run([accuracy,merged], feed_dict={x: batch_xt, y_: batch_yt})
    avg_acc += c
print('Inference Time:', time.time() -start_time)
print('Test Accuracy: ', avg_acc/total_batch)
print('Pre: ', avg_acc_pre/total_batch)

for variable in tf.trainable_variables():
    print ('variable:', variable)

saved_model_name = save_path + "/model.ckpt.quantized" + str(quantization_bits)
saver.save(sess, saved_model_name)

total_parameters = 0
for variable in tf.trainable_variables():
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    total_parameters += variable_parameters

print( 'No of parameters in the model is :',total_parameters)
