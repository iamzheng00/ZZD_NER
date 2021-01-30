import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow_addons.text import crf
from DataUtils import maxlen
import time



class conf():
    LSTM_dim=100
    tag_num=18

class Model_bilstm(keras.Model):
    def __init__(self, conf, ):
        super(Model_bilstm, self).__init__()
        # 模型基本参数
        self.LSTM_dim = conf.LSTM_dim
        self.tag_num = conf.tag_num
        # 模型所需的层定义
        self.embedding = layers.Embedding(input_dim=10000,output_dim=100, mask_zero=True)

        self.dense = layers.Dense(self.tag_num)
        self.fw_LSTM = layers.LSTM(units=self.LSTM_dim, return_sequences=True, go_backwards=False)
        self.bw_LSTM = layers.LSTM(units=self.LSTM_dim, return_sequences=True, go_backwards=True)
        self.BiLSTM = layers.Bidirectional(self.fw_LSTM, backward_layer=self.bw_LSTM)

    def call(self, inputs, training=None, mask=None):
        x = self.embedding(inputs)
        x = self.BiLSTM(x)
        x = self.dense(x)

        return x

    def crf_loss(self,input,tag_ids,sentence_len_list):
        loss, self.trans_p = crf.crf_log_likelihood(inputs=input, tag_indices=tag_ids,sequence_lengths=sentence_len_list)
        return tf.reduce_sum(loss)

# 从批数据中 分离出seq_ids 和 tag_ids,以及每一句的长度列表：seq_len_list
def get_train_data_from_batch(batch):
    seq_ids, tag_ids = [],[]
    for sentence in batch:
        seq_ids.append(sentence[1])
        tag_ids.append(sentence[3])
    seq_len_list = [len(s) for s in seq_ids]
    seq_ids = keras.preprocessing.sequence.pad_sequences(seq_ids,padding='post',value=0)
    tag_ids = keras.preprocessing.sequence.pad_sequences(tag_ids,padding='post',value=0)
    seq_ids = tf.convert_to_tensor(seq_ids,dtype='int32')
    tag_ids = tf.convert_to_tensor(tag_ids,dtype='int32')
    return (seq_ids,tag_ids,seq_len_list)

def train_one_epoch(mymodel, batches,epoch_num=1):
    '''

    :param mymodel:
    :param batches: one batch data: [[sentence],[sentence],....]
                           sentence=[[chars],[charids],[tags],[tag_ids]]
    :param epoch_num:
    :return:
    '''
    optimizer = tf.optimizers.Adam(learning_rate=.0005)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=mymodel)
    manager = tf.train.CheckpointManager(checkpoint, directory='checkpoints', max_to_keep=5)

    # =====run model=======
    for batch_num, batch in enumerate(batches):
        seq_ids, tag_ids, seq_len_list= get_train_data_from_batch(batch)
        with tf.GradientTape() as tape:
            logits = mymodel(seq_ids)
            loss=mymodel.crf_loss(logits,tag_ids,seq_len_list)
            seq_tags, best_score = crf.crf_decode(potentials=logits,transition_params=mymodel.trans_p, sequence_length=seq_len_list)
        grads = tape.gradient(loss, mymodel.trainable_variables)
        optimizer.apply_gradients(zip(grads,mymodel.trainable_variables))
        # optimizer.minimize(loss, [myModel_bilstm.trainable_variables])
        if batch_num%50==10:
            manager.save(checkpoint_number=batch_num+epoch_num*len(batches))



if __name__ == '__main__' :
    from DataUtils import read_train_data,get_batches

    starttime=time.time()
    train_data_path = r'F:\zzd\毕业论文\论文代码\NER\data\someNEWS_BIOES.dev'
    vocab_path = r'F:\zzd\毕业论文\论文代码\NER\vocab\vocab.pkl'
    data = read_train_data(train_data_path, vocab_path)
    batches = get_batches(data, 100)
    # print(batches[1])
    configers = conf()
    model = Model_bilstm(configers)
    train_one_epoch(model, batches,1)
    endtime = time.time()
    ptime = endtime-starttime
    print('done!',str(ptime))

