import utils as util
import read_pkl as load
import common_config as config
import preprocess as preprocess
import utils as utils
import pdb
import numpy as np
import tensorflow as tf

def test_cyclic_lr():
    LR_low = 0
    LR_high = 10
    step_size = 20
    train_batch=40
    for iteration in range(0,train_batch):
        lr = util.get_triangular_lr(iteration, step_size, LR_low, LR_high)
        print lr

def test_embedding_compression(k):
    data_path = '/home/ec2-user/Compression_exp/datasets/alexa/'
    train_df, dev_df, test_df = load.extract()
    prepro = preprocess.PROCESSED_DATA(train_df["sentence"],train_df["label"])
    word_to_index, index_to_embedding = utils.load_embedding_from_disks_dense(config.EMBED_FILE, prepro.word2index)
    #U,s,VT = np.linalg.svd(index_to_embedding, full_matrices=True)
    #tf_config = tf.ConfigProto(log_device_placement=True)
    #tf_config.gpu_options.allow_growth=True
    #sess = tf.Session(config=tf_config)
    St,Ut,Vt = tf.svd(index_to_embedding)
    #p = 1 - p
    #k = int(p * (index_to_embedding.shape[0]*index_to_embedding.shape[1]/(index_to_embedding.shape[0]+index_to_embedding.shape[1])))
    Sk = np.diag(St[0:k])
    Uk = Ut[:, 0:k]
    Vk = Vt[0:k, :]
    pdb.set_trace()
    print('Blah')
    return Uk
def main():
    k = 50
    Uk = test_embedding_compression(k)
if __name__ == "__main__":
    main()
