from common_config import *
from pretrain_config import *
from network import *
from initialize import *
from utils import *
import alexa_extract as alexa
import imdb_extract as imdb
import read_pkl as load
import preprocess as preprocess

import time
import pandas as pd
import numpy as np
import pdb

### ------  Pre-Process and load Data ---- ###
start_time = time.time()
training_type = 'PreTrain'
#assert os.path.exists(pretrained_model_path+'.meta')
if EXP =='alexa':
    save_path = model_path + EXP + '/'+ domain + '/' + training_type + '/' +  netType
else:
    save_path = model_path + EXP + '/'+ training_type + '/' +  netType

if not os.path.exists(save_path):
    os.makedirs(save_path + '/train')
    os.makedirs(save_path + '/test')

if EXP =='alexa':
    print("Preparing Large Alexa Data")
    train_df, dev_df, test_df = alexa.extract()
elif EXP =='IMDB':
    print("Preparing Large MR Data")
    train_df, dev_df, test_df = imdb.extract()
else:
    print('Preparing Data For: ',EXP)
    train_df, dev_df, test_df = load.extract()

preprocess = preprocess.PROCESSED_DATA(train_df["sentence"],train_df["label"])
print ('Time to Prepare Data:' ,time.time()-start_time)

if netType == 'DAN' or netType =='DAN_LR' or netType=='LSTM' or netType=='BiLSTM' or netType=='GRU':
    start_time = time.time()
    print('Creating Vocab From Glove')
    word_to_index, index_to_embedding = load_embedding_from_disks_dense(EMBED_FILE, preprocess.word2index)
    print ('Time to process Glove Embedding :' ,time.time()-start_time)
    pad_val = word_to_index['_WORD_NOT_FOUND']

if Emb_Compress == 'true':
        p = 1 - compression
        k = int(p*EMBEDDING_DIM)
        EMBEDDING_DIM = k
    
#import pdb; pdb.set_trace()
## -------------- TF Graph  --------------- ###
## Define and Initialize Network
Training_Time = 0
Inference_Time = 0
tf.reset_default_graph()
tf.set_random_seed(0)
if netType == 'MLP':
    weights, biases = initialize_nw(netType, preprocess.VOCAB_SIZE, preprocess.NUM_CLASSES)
    x = tf.placeholder(tf.float32, [None, preprocess.VOCAB_SIZE])
    y = MLP(x,weights,biases)
    y_ = tf.placeholder(tf.float32, [None, preprocess.NUM_CLASSES])
    regularizer = tf.nn.l2_loss(weights['w1']) + tf.nn.l2_loss(weights['w2']) + tf.nn.l2_loss(weights['w3']) + tf.nn.l2_loss(weights['w4'])

elif netType == 'MLP_LR':
    weights, biases = initialize_nw(netType, preprocess.VOCAB_SIZE, preprocess.NUM_CLASSES)
    x = tf.placeholder(tf.float32, [None, preprocess.VOCAB_SIZE])
    y = MLP_LR(x,weights,biases)
    y_ = tf.placeholder(tf.float32, [None, preprocess.NUM_CLASSES])
    regularizer = tf.nn.l2_loss(weights['w11']) + tf.nn.l2_loss(weights['w12']) + tf.nn.l2_loss(weights['w2']) + tf.nn.l2_loss(weights['w3'])

elif netType == 'DAN':
    vocab_size = index_to_embedding.shape[0]
    weights, biases = initialize_nw(netType, vocab_size,preprocess.NUM_CLASSES+1,EMBEDDING_DIM)
    x = tf.placeholder(tf.int32, [None, None])
    embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, EMBEDDING_DIM])
    y = DAN(x,weights,biases)
    y_ = tf.placeholder(tf.float32, [None, preprocess.NUM_CLASSES +1])
    #regularizer = tf.nn.l2_loss(weights['w1']) + tf.nn.l2_loss(weights['w2']) + tf.nn.l2_loss(weights['w3']) + tf.nn.l2_loss(weights['w4'])

elif netType == 'DAN_LR':
    vocab_size = index_to_embedding.shape[0]
    weights, biases = initialize_nw(netType, vocab_size, preprocess.NUM_CLASSES+1)
    x = tf.placeholder(tf.int32, [None, None])
    p = 1 - Percentage_Reduction
    k = int(p * (vocab_size * EMBEDDING_DIM/(vocab_size + EMBEDDING_DIM)))
    embeddings = tf.Variable(tf.random_uniform([vocab_size, k], -1.0, 1.0))
    embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, k])
    y = DAN_LR(x,weights,biases)
    y_ = tf.placeholder(tf.float32, [None, preprocess.NUM_CLASSES +1])
    #regularizer = tf.nn.l2_loss(weights['w11']) + tf.nn.l2_loss(weights['w12']) + tf.nn.l2_loss(weights['w2']) + tf.nn.l2_loss(weights['w3']) + tf.nn.l2_loss(weights['w4'])

elif netType =='LSTM':
    vocab_size = index_to_embedding.shape[0]
    weights, biases = initialize_nw(netType, vocab_size,preprocess.NUM_CLASSES+1,EMBEDDING_DIM)
    x = tf.placeholder(tf.int32, [None, None])
    embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, EMBEDDING_DIM])
    y = LSTM(x,weights,biases)
    y_ = tf.placeholder(tf.float32, [None, preprocess.NUM_CLASSES +1])
    regularizer = tf.nn.l2_loss(weights['w1']) + tf.nn.l2_loss(weights['w2']) #+ tf.nn.l2_loss(weights['w3']) + tf.nn.l2_loss(weights['w4'])

elif netType =='BiLSTM':
    vocab_size = index_to_embedding.shape[0]
    weights, biases = initialize_nw(netType, vocab_size, preprocess.NUM_CLASSES+1)
    x = tf.placeholder(tf.int32, [None, None])
    embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, EMBEDDING_DIM])
    y = BiLSTM(x,weights,biases)
    y_ = tf.placeholder(tf.float32, [None, preprocess.NUM_CLASSES +1])

elif netType =='GRU':
    vocab_size = index_to_embedding.shape[0]
    weights, biases = initialize_nw(netType, vocab_size, preprocess.NUM_CLASSES+1)
    x = tf.placeholder(tf.int32, [None, None])
    embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, EMBEDDING_DIM])
    y = GRU(x,weights,biases)
    y_ = tf.placeholder(tf.float32, [None, preprocess.NUM_CLASSES +1])
    regularizer = tf.nn.l2_loss(weights['w1']) + tf.nn.l2_loss(weights['w2']) + tf.nn.l2_loss(weights['w3']) + tf.nn.l2_loss(weights['w4'])


# Cost Function
#pdb.set_trace()
CE = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)

cross_entropy = tf.reduce_mean(CE)
tf.summary.scalar('cross_entropy', cross_entropy)
if REGULARIZATION =='true':
    loss = tf.reduce_mean(cross_entropy + REGULARIZATION_FRACTION * regularizer)
else:
    loss = cross_entropy

## Optimization
#train_step = tf.train.AdagradOptimizer(LR).minimize(loss)
#global_step = tf.Variable(0, trainable=False)
#learning_rate = tf.train.exponential_decay(LR_0, global_step,100000, 0.96, staircase=True)
#optimizer=tf.train.RMSPropOptimizer(LR_0)
#optimizer = tf.train.AdamOptimizer(LR_0)
#LR = LR_0
LR_0 = tf.placeholder(tf.float32, [], name='learning_rate')
if optimize == 'SGD':
    optimizer = tf.train.GradientDescentOptimizer(LR_0)
elif optimize == 'ADAM':
    optimizer = tf.train.AdamOptimizer(learning_rate=LR_0, beta1=beta1,beta2=beta2,epsilon=epsilon)
elif optimize == 'RMSProp':
    optimizer = tf.train.RMSPropOptimizer(LR_0)
elif optimize == 'ADAGRAD':
    optimizer = tf.train.AdagradOptimizer(LR_0)

train_step = optimizer.minimize(loss)

## Evaluation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

### ----- Session Configs etc --------- ###
## Saving and Starting Session
saver = tf.train.Saver(max_to_keep=100)
init = tf.global_variables_initializer()

## GPU Configurations
config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth=True

sess = tf.Session(config=config)
sess.run(init)
if netType == 'DAN' or netType== 'LSTM':
    embedding_init = weights['w1'].assign(embedding_placeholder)
    if Emb_Compress == 'true':
        St,Ut,Vt = sess.run(tf.svd(index_to_embedding))
        Uk = Ut[:, 0:k]
        index_to_embedding = Uk
    sess.run(embedding_init,feed_dict={embedding_placeholder: index_to_embedding})

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(save_path + '/train',sess.graph)
test_writer = tf.summary.FileWriter(save_path + '/test')

### ------------ Run Epochs -------------###
Total_Epoch = EPOCH-STARTING_EPOCH
f = open(save_path + '/' + 'Learning Schedule','w+')
f = open(save_path + '/' + 'dev_acc','w+')
best_dev_acc = 0
best_test_acc = 0
best_model_name = save_path + "/save/model.ckpt.best"
best_model_test = save_path + "/save/model.ckpt.best_test"
INF_t=0
INF_n=0
iteration=0
for epoch in range(STARTING_EPOCH,EPOCH):
    ### Learning Rate and Batch Linear Schedule
    if epoch%freq==0 and epoch!=STARTING_EPOCH:
        LR_high = LR_high * decay
    if LR_high <= LR_low:
        LR_high = LR_high_initial
        LR_low =  LR_high_initial* math.pow(decay , 5)

    avg_cost = 0
    avg_acc = 0

    start_model_name = save_path + "/model.ckpt." + str(epoch)
    saved_model_name = save_path + "/model.ckpt." + str(epoch+1)

    train_sample_size=len(train_df["sentence"])
    train_batch = int(train_sample_size / BATCH_SIZE )
    dev_sample_size=len(dev_df["sentence"])
    dev_batch = int(dev_sample_size / BATCH_SIZE )
    test_sample_size=len(test_df["sentence"])
    test_batch = int(test_sample_size / BATCH_SIZE )

    try:
        saver.restore(sess, start_model_name)
        print ("loaded model: ", start_model_name)
    except:
        print ("no loaded model: starting Training from scratch")
    print("----------------Epoch",'%04d' % (epoch+1)," Started ----------------")
    print ('Learning Rate Upper Bound: => ', LR_high)
    print ('Learning Rate Lowe Bound: => ', LR_low)

    print('Vocab Size: ', preprocess.VOCAB_SIZE)
    print("Training on Batches of size,", '%04d' %(BATCH_SIZE))
    start_time = time.time()
    for i in range(train_batch):
        iteration+=1
        if netType =='DAN' or netType=='DAN_LR' or netType=='LSTM' or netType=='GRU':
            batch_xs, batch_ys = preprocess.get_batch_dense(train_df["sentence"], train_df["label"], i, BATCH_SIZE, word_to_index)
            batch_xs = pad_maxlen(batch_xs, pad_val)
            f = open(save_path + '/' + 'Learning_Schedule','a+')
        else:
            batch_xs, batch_ys = preprocess.get_batch_sparse(train_df["sentence"], train_df["label"], i, BATCH_SIZE)

        LR = LR_high
        if LR_Schedule == 'cyclic':
            LR = get_triangular_lr(iteration, step_size, LR_low, LR_high)
        c,_, a, summary,lr = sess.run([cross_entropy,train_step,accuracy,merged,LR_0], feed_dict={x: batch_xs, y_: batch_ys, LR_0: LR})
        if iteration%100==0:
            batch_xd, batch_yd = preprocess.get_dense(dev_df["sentence"], dev_df["label"], word_to_index)
            batch_xd = pad_maxlen(batch_xd, pad_val)
            dev_a = sess.run(accuracy, feed_dict={x: batch_xd, y_: batch_yd})
            print("iteration:", iteration, 'LR' , lr, ' DEV ACC:', dev_a)
            f = open(save_path + '/' + 'dev_acc','a+')
            f.write("iteration:"+ str(iteration) + ' DEV ACC:'+ str(dev_a) +'\n')
            batch_xt, batch_yt = preprocess.get_dense(test_df["sentence"], test_df["label"], word_to_index)
            batch_xt = pad_maxlen(batch_xt, pad_val)
            test_a = sess.run(accuracy, feed_dict={x: batch_xt, y_: batch_yt})
            f.write("iteration:"+ str(iteration) + ' DEV ACC:'+ str(dev_a) +'\n')
            if(test_a > best_test_acc):
                print('--------- Best Test Model Update ------')
                best_test_acc = test_a #avg_acc/test_batch
                saver.save(sess, best_model_test)

            print("iteration:", iteration, 'LR' , lr, ' TEST ACC:', test_a)
        avg_cost += c
        avg_acc += a
        train_writer.add_summary(summary, iteration)
    Training_Time += time.time()-start_time
    print ("Saving Model:", saved_model_name)
    saver.save(sess, saved_model_name)
    print ("Epoch:", '%04d' % (epoch+1), "loss=","{:.9f}".format(avg_cost / train_batch))
    print ("Epoch:", '%04d' % (epoch+1), "Training Acc =","{:.9f}".format(avg_acc/train_batch))
    start_time=time.time()
    batch_xd, batch_yd = preprocess.get_dense(dev_df["sentence"], dev_df["label"], word_to_index)
    batch_xd = pad_maxlen(batch_xd, pad_val)
    dev_a = sess.run(accuracy, feed_dict={x: batch_xd, y_: batch_yd})
    batch_xt, batch_yt = preprocess.get_dense(test_df["sentence"], test_df["label"], word_to_index)
    batch_xt = pad_maxlen(batch_xt, pad_val)
    inf_t = time.time()
    test_a = sess.run(accuracy, feed_dict={x: batch_xt, y_: batch_yt})
    print('------- Inference Time For -----', time.time() - inf_t)
    
    INF_n+=1
    f.write("iteration:"+ str(iteration) + ' DEV ACC:'+ str(dev_a) +'\n')
    if(dev_a > best_dev_acc):
        print('--------- Best Dev Model Update ------')
        best_dev_acc = dev_a # avg_acc/dev_batch
        saver.save(sess, best_model_name)
    f = open(save_path + '/' + 'results','a+')
    print("Epoch:", '%04d' % (epoch+1), "Dev-Set_Acc is: ", dev_a) #avg_acc / dev_batch)
    f.write('-----------------\n')
    f.write('Dev Set Results for:  ' + netType +': ' + training_type + '\n')
    f.write('-----------------\n')
    f.write('-----------------\n')
    f.write("Epoch:"+ str(epoch+1) + "Dev_Acc is: " +str(dev_a)+'\n')
    f.write('        ')
    f.close()
    if(test_a > best_test_acc):
        print('--------- Best Test Model Update ------')
        best_test_acc = test_a #avg_acc/test_batch
        saver.save(sess, best_model_test)
    print("Epoch:", '%04d' % (epoch+1),"Test-Set_Acc is: ", test_a)
    print("----------------Epoch Finished ----------------")
    print(" ")
total_parameters = 0
for variable in tf.trainable_variables():
    print('Variable:' , variable)
    shape = variable.get_shape()
    print(shape)
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    total_parameters += variable_parameters
print( 'No of parameters in the model is :',total_parameters)
print('BEST TEST ACC: ', best_test_acc)
print('Average Trainign Time Per Epoch: ', Training_Time/Total_Epoch )
print('Average Inference Time: ', INF_t/INF_n)
sess.close()
