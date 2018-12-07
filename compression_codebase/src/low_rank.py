from common_config import *
from finetune_config import *
from network import *
import initialize as init
from utils import *
import alexa_extract as alexa
import imdb_extract as imdb
import read_pkl as load
import preprocess as preprocess

import time
import pandas as pd
import numpy as np
import pdb
import sys

"""
finetune.py [percentage reduction]
"""


### ------  Pre-Process and load Data ---- ###
start_time = time.time()
training_type = 'FineTune'                 # Options: 'PreTrain', 'FineTune', 'Quantization'

## -----For FineTune and Quantization ::
## For FineTuning The Model to Compress Can be Specified here
Percentage_Reduction = float(sys.argv[1]) #0.7
pretrain_model = 'model.ckpt.best_test'

    #assert os.path.exists(pretrained_model_path+'.meta')
if EXP =='alexa':
    save_path = model_path + EXP + '/'+ domain + '/' + training_type + '/' +  netType + str(Percentage_Reduction)
    pretrained_model_path = model_path + EXP + '/' + domain + '/'+ 'PreTrain/' + netType + '/save/' + pretrain_model#model.ckpt.67'
    #assert os.path.exists(pretrained_model_path+'.meta')
else:
    save_path = model_path + EXP + '/'+ training_type + '/' +  netType + str(Percentage_Reduction)
    pretrained_model_path = model_path + EXP + '/' + 'PreTrain/' + netType + '/save/' + pretrain_model

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

#import pdb; pdb.set_trace()
## -------------- TF Graph  --------------- ###
## Define and Initialize Network
Training_Time = 0
Inference_Time = 0
tf.reset_default_graph()
if netType == 'DAN':
    vocab_size = index_to_embedding.shape[0]
    weights, biases = init.initialize_nw(netType, vocab_size, preprocess.NUM_CLASSES+1)
    x = tf.placeholder(tf.int32, [None, None])
    embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, EMBEDDING_DIM])
    y = DAN(x,weights,biases)
    y_ = tf.placeholder(tf.float32, [None, preprocess.NUM_CLASSES +1])
    regularizer = tf.nn.l2_loss(weights['w1']) + tf.nn.l2_loss(weights['w2']) + tf.nn.l2_loss(weights['w3']) + tf.nn.l2_loss(weights['w5'])

elif netType =='LSTM':
    vocab_size = index_to_embedding.shape[0]
    weights, biases = init.initialize_nw(netType, vocab_size, preprocess.NUM_CLASSES+1)
    x = tf.placeholder(tf.int32, [None, None])
    embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, EMBEDDING_DIM])
    y = LSTM(x,weights,biases)
    y_ = tf.placeholder(tf.float32, [None, preprocess.NUM_CLASSES +1])
    regularizer = tf.nn.l2_loss(weights['w1']) + tf.nn.l2_loss(weights['w2']) #+ tf.nn.l2_loss(weights['w3']) + tf.nn.l2_loss(weights['w4'])

# Cost Function
CE = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
cross_entropy = tf.reduce_mean(CE)
tf.summary.scalar('cross_entropy', cross_entropy)
if REGULARIZATION =='true':
    loss = tf.reduce_mean(cross_entropy + REGULARIZATION_FRACTION * regularizer)
else:
    loss = cross_entropy
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
    sess.run(embedding_init,feed_dict={embedding_placeholder: index_to_embedding})

merged = tf.summary.merge_all()

### ------------ Run Epochs -------------###
Total_Epoch = EPOCH-STARTING_EPOCH

#pretrained_model_path = save_path +"/save/model.ckpt.best_test"
saver.restore(sess, pretrained_model_path)
print ("Loaded Model Performing SVD:", pretrained_model_path)
#p =sys.argv[1]
BATCH_SIZE = BATCH_SIZE

train_sample_size=len(train_df["sentence"])
train_batch = int(train_sample_size / BATCH_SIZE )
dev_sample_size=len(dev_df["sentence"])
#DEV_BATCH_SIZE=dev_sample_size
dev_batch = int(dev_sample_size/BATCH_SIZE)

test_sample_size=len(test_df["sentence"])
test_batch = int(test_sample_size / BATCH_SIZE )

if netType == 'LSTM':
    w1 = sess.run('W1:0')
    w2 = sess.run('W2:0')
    b2 = sess.run('b2:0')

    ### Svd Step - Create a Separate Function
    p = 1 - Percentage_Reduction
    k = int(p * (w1.shape[0]*w1.shape[1]/(w1.shape[0]+w1.shape[1])))
    start_time =time.time()
    St,Ut,Vt = sess.run(tf.svd(weights['w1']))
    Sk = np.diag(St[0:k])
    Uk = Ut[:, 0:k]
    Vk = Vt[0:k, :]
    eig = open(save_path + '/' + 'EigenValues','a+')
    eig.write(St)
    eig.close()
    # Low Rank Weights
    w11 = Uk
    w12 = sess.run(tf.matmul(Sk,Vk))

    tf.reset_default_graph()

    vocab_size = w11.shape[0]
    embed_dim = w11.shape[1]
    weights = {
        'w11': tf.Variable(tf.constant(0.0, shape=[vocab_size, embed_dim]),trainable=True, name="W11"),
        'w12': tf.Variable(w12, name="W12"),
        'w2': tf.Variable(w2,name = 'W2')
    }
    biases = {
        'b2': tf.Variable(b2,name = 'b2')
    }
    #x = tf.placeholder(tf.int32, [None, None])
    x = tf.placeholder(tf.int32, [BATCH_SIZE, maxlen])
    embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, k])
    y = LSTM_LR(x,weights,biases)
    y_ = tf.placeholder(tf.float32, [None, preprocess.NUM_CLASSES +1])

elif netType == 'DAN':
    w1 = sess.run('W1:0')
    w2 = sess.run('W2:0')
    w3 = sess.run('W3:0')
    w5 = sess.run('W5:0')
    b2 = sess.run('b2:0')
    b3 = sess.run('b3:0')
    b5 = sess.run('b5:0')
    ### Svd Step - Create a Separate Function
    p = 1 - Percentage_Reduction
    k = int(p * (w1.shape[0]*w1.shape[1]/(w1.shape[0]+w1.shape[1])))
    start_time =time.time()
    St,Ut,Vt = sess.run(tf.svd(weights['w1']))
    Sk = np.diag(St[0:k])
    Uk = Ut[:, 0:k]
    Vk = Vt[0:k, :]
    eig = open(save_path + '/' + 'EigenValues','a+')
    eig.write(St)
    eig.close()
    # Low Rank Weights
    w11 = Uk
    w12 = sess.run(tf.matmul(Sk,Vk))

    tf.reset_default_graph()

    vocab_size = w11.shape[0]
    embed_dim = w11.shape[1]
    weights = {
        'w11': tf.Variable(tf.constant(0.0, shape=[vocab_size, embed_dim]),trainable=True, name="W11"),
        'w12': tf.Variable(w12, name="W12"),
        'w2': tf.Variable(w2,name = 'W2'),
        'w3': tf.Variable(w3,name = 'W3'),
        'w5': tf.Variable(w5,name = 'W5')
    }
    biases = {
        'b2': tf.Variable(b2,name = 'b2'),
        'b3': tf.Variable(b3,name = 'b3'),
        'b5': tf.Variable(b5,name = 'b5')
    }
    #x = tf.placeholder(tf.int32, [None, None])
    x = tf.placeholder(tf.int32, [BATCH_SIZE, maxlen])
    embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, k])
    y = DAN_LR(x,weights,biases)
    y_ = tf.placeholder(tf.float32, [None, preprocess.NUM_CLASSES +1])

# Cost Function
CE = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
cross_entropy = tf.reduce_mean(CE)
tf.summary.scalar('cross_entropy', cross_entropy)
regularizer = tf.nn.l2_loss(weights['w11']) + tf.nn.l2_loss(weights['w12']) + tf.nn.l2_loss(weights['w2'])
if REGULARIZATION =='true':
    loss = tf.reduce_mean(cross_entropy + REGULARIZATION_FRACTION * regularizer)
else:
    loss = cross_entropy

LR_0 = tf.placeholder(tf.float32, [], name='learning_rate')
if finetune_optimize == 'SGD':
    optimizer = tf.train.GradientDescentOptimizer(LR_0)
elif finetune_optimize == 'ADAM':
    optimizer = tf.train.AdamOptimizer(learning_rate=LR_0, beta1=beta1,beta2=beta2,epsilon=epsilon)
elif finetune_optimize == 'RMSProp':
    optimizer = tf.train.RMSPropOptimizer(LR_0)
elif finetune_optimize == 'ADAGRAD':
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
    embedding_init = weights['w11'].assign(embedding_placeholder)
    sess.run(embedding_init,feed_dict={embedding_placeholder: w11})

best_dev_acc = 0
best_test_acc = 0
best_model_name = save_path + "/save/model.ckpt.low_rank.best"
best_model_test = save_path +"/save/model.ckpt.low_rank.best_test"

f = open(save_path + '/' + 'dev_acc','a+')
iteration=0
for epoch in range(STARTING_EPOCH,EPOCH):
    ### Learning Rate Schedule
    if epoch%freq==0 and epoch!=STARTING_EPOCH:
        LR_high = LR_high * decay
    if LR_high <= LR_low:
        LR_high = LR_high_initial
        LR_low =  LR_high* math.pow(decay , 5)
    avg_cost = 0
    avg_acc = 0

    start_model_name = save_path + "/model.ckpt.low_rank" + str(epoch)
    saved_model_name = save_path + "/model.ckpt.low_rank" + str(epoch+1)
    
    train_sample_size=len(train_df["sentence"])
    train_batch = int(train_sample_size / BATCH_SIZE )
    dev_sample_size=len(dev_df["sentence"])
    dev_batch = int(dev_sample_size / BATCH_SIZE )
    test_sample_size=len(test_df["sentence"])
    test_batch = int(test_sample_size / BATCH_SIZE )

    print("----------------Epoch",'%04d' % (epoch+1)," Started ----------------")
    try:
        saver.restore(sess, start_model_name)
        print ("loaded model: ", start_model_name)
    except:
        print ("no loaded model: starting Training from scratch")
    print ('Learning Rate UB: => ', LR_high)
    print ('Learning Rate LB: => ', LR_low)
    print('Vocab Size: ', preprocess.VOCAB_SIZE)
    print("Training on Batches of size,", '%04d' %(BATCH_SIZE))
    start_time = time.time()
    for i in range(train_batch):
        iteration+=1
        if netType =='DAN' or netType=='DAN_LR' or netType=='LSTM' or netType=='GRU':
            batch_xs, batch_ys = preprocess.get_batch_dense(train_df["sentence"], train_df["label"], i, BATCH_SIZE, word_to_index)
            batch_xs = pad_maxlen(batch_xs, pad_val)
        else:
            batch_xs, batch_ys = preprocess.get_batch_sparse(train_df["sentence"], train_df["label"], i, BATCH_SIZE)
        LR = LR_high
        if LR_Schedule == 'cyclic':
            LR = get_triangular_lr(iteration, step_size, LR_low, LR_high)
        c_train,_, a,lr = sess.run([cross_entropy,train_step,accuracy,LR_0], feed_dict={x: batch_xs, y_: batch_ys, LR_0:LR})
        avg_cost += c_train
        avg_acc += a

        if iteration%100==0:
            dev_acc = 0
            for j in range(dev_batch):
                if netType == 'DAN' or netType =='DAN_LR' or netType=='LSTM' or netType == 'GRU':
                    try:
                        batch_xd, batch_yd = preprocess.get_batch_dense(dev_df["sentence"], dev_df["label"], j, BATCH_SIZE, word_to_index)
                        batch_xd = pad_maxlen(batch_xd, pad_val)
                        c = sess.run(accuracy, feed_dict={x: batch_xd, y_: batch_yd})
                    except:
                        pass
                else:
                    try:
                        batch_xd, batch_yd = preprocess.get_batch_sparse(dev_df["sentence"], dev_df["label"], j, BATCH_SIZE)
                        c = sess.run(accuracy, feed_dict={x: batch_xd, y_: batch_yd})
                    except:
                        pass
                dev_acc += c
            dev_a = dev_acc/dev_batch
            print("iteration:", iteration, 'learning rate=',lr,' DEV ACC:', dev_a)
            f.write("iteration:"+ str(iteration) + ' DEV ACC:'+ str(dev_a) +'\n')
            test_acc = 0
            for i in range(test_batch):
                if netType == 'DAN' or netType =='DAN_LR' or netType=='LSTM':
                    try:
                        batch_xt, batch_yt = preprocess.get_batch_dense(test_df["sentence"], test_df["label"], i, BATCH_SIZE, word_to_index)
                        batch_xt = pad_maxlen(batch_xt, pad_val)
                        c = sess.run(accuracy, feed_dict={x: batch_xt, y_: batch_yt})
                    except:
                        pass
                else:
                    try:
                        batch_xt, batch_yt = preprocess.get_batch_sparse(dev_df["sentence"], test_df["label"], i, BATCH_SIZE)
                        c = sess.run(accuracy, feed_dict={x: batch_xt, y_: batch_yt})
                    except:
                        pass
                test_acc += c
            test_a = test_acc/test_batch
            if(test_a > best_test_acc):
                print('--------- Best Test Model Update ------')
                best_test_acc = test_a #avg_acc/test_batch
                saver.save(sess, best_model_test)
            print("iteration:", iteration, 'learning rate=',lr,' TEST ACC:',
                    test_a)
            f.write("iteration:"+ str(iteration) + ' TEST ACC:'+ str(test_a) +'\n')
    Training_Time += time.time()-start_time
    print ("Time for Training Epoch:" , epoch+1, "is:  ", time.time()-start_time )
    print ("Saving Model:", saved_model_name)
    saver.save(sess, saved_model_name)
    print ("Epoch:", '%04d' % (epoch+1), "loss=","{:.9f}".format(avg_cost / train_batch))
    print ("Epoch:", '%04d' % (epoch+1), "Training Acc =","{:.9f}".format(avg_acc/train_batch))

    if(dev_a > best_dev_acc):
        print('--------- Best Dev Model Update ------')
        best_dev_acc = dev_a #avg_acc/dev_batch
        saver.save(sess, best_model_name)
    print("Epoch:", '%04d' % (epoch+1), "Dev-Set_Acc is: ", dev_a)
    test_acc = 0
    inf_time = 0
    for i in range(test_batch):
        if netType == 'DAN' or netType =='DAN_LR' or netType=='LSTM':
            try:
                batch_xt, batch_yt = preprocess.get_batch_dense(test_df["sentence"], test_df["label"], i, BATCH_SIZE, word_to_index)
                #batch_xt = numpy_pad(batch_xt, pad_val)
                batch_xt = pad_maxlen(batch_xt, pad_val)
                strart_time = time.time()
                c = sess.run(accuracy, feed_dict={x: batch_xt, y_: batch_yt})
                inf_time += time.time() - start_time
            except:
                pass
        else:
            try:
                strart_time = time.time()
                batch_xt, batch_yt = preprocess.get_batch_sparse(dev_df["sentence"], test_df["label"], i, BATCH_SIZE)
                c = sess.run(accuracy, feed_dict={x: batch_xt, y_: batch_yt})
                inf_time += time.time() - start_time
            except:
                pass
        test_acc += c
    test_a = test_acc/test_batch
    if(test_a > best_test_acc):
        print('--------- Best Test Model Update ------')
        best_test_acc = test_a #avg_acc/test_batch
        saver.save(sess, best_model_test)
 
    print("Epoch:", '%04d' % (epoch+1),"Test-Set_Acc is: ", test_a)
    print ("Time for Inference is:  ", inf_time )
    print("----------------Epoch Finished ----------------")
    print(" ")

total_parameters = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    print('Variable:' , variable)
    shape = variable.get_shape()
    print(shape)
    #print(len(shape))
    variable_parameters = 1
    for dim in shape:
     #   print(dim)
        variable_parameters *= dim.value
    #print(variable_parameters)
    total_parameters += variable_parameters
print( 'No of parameters in the model is :',total_parameters)
#f.write('Test Results for:' % netType)
#f.write("Epoch:", '%04d' % (epoch+1), "Test_Acc is: ", avg_acc / total_batch)

#f.write('Vocab Size:' , preprocess.VOCAB_SIZE)
#f.write('Number of Classes:', preprocess.NUM_CLASSES)
#f.write('        ')
print('BEST TEST ACC: ', best_test_acc)
print('Average Trainign Time Per Epoch: ', Training_Time/Total_Epoch )
print('Average Inference Time: ', Inference_Time/Total_Epoch)
sess.close()

