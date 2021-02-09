import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow_addons.text import crf
from DataUtils2 import maxlen, findall_tag, P_R_F1_score
import time


class conf():
    LSTM_dim = 100
    tag_num = 18


class Model_bilstm(keras.Model):
    def __init__(self, conf, ):
        super(Model_bilstm, self).__init__()
        # 模型基本参数
        self.LSTM_dim = conf.LSTM_dim
        self.tag_num = conf.tag_num
        # 模型所需的层定义
        self.embedding = layers.Embedding(input_dim=10000, output_dim=100, mask_zero=True)
        self.fw_LSTM = layers.LSTM(units=self.LSTM_dim, return_sequences=True, go_backwards=False)
        self.bw_LSTM = layers.LSTM(units=self.LSTM_dim, return_sequences=True, go_backwards=True)
        self.BiLSTM = layers.Bidirectional(self.fw_LSTM, backward_layer=self.bw_LSTM)
        self.dense = layers.Dense(self.tag_num)

    def call(self, inputs, training=None, mask=None):
        x = self.embedding(inputs)
        x = self.BiLSTM(x)
        x = self.dense(x)
        return x

    def crf_loss(self, input, tag_ids, sentence_len_list):
        loss, self.trans_p = crf.crf_log_likelihood(inputs=input, tag_indices=tag_ids,
                                                    sequence_lengths=sentence_len_list)
        return tf.reduce_mean(loss)



def train_one_epoch(mymodel, charid_batches,tagid_batches,seq_len_batches, epoch_num=1):
    '''

    :param mymodel:
    :param charid_batches: [batch_size, padded_sentence_length]
    :param tagid_batches: [batch_size, padded_sentence_length]
    :param seq_len_batches: [batch_size, sentence_length]
    :param epoch_num:
    :return:
    '''
    optimizer = tf.optimizers.Adam(learning_rate=.005)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=mymodel)
    manager = tf.train.CheckpointManager(checkpoint, directory='checkpoints01', max_to_keep=10)

    # =====run model=======
    for batch_num, seq_ids in enumerate(charid_batches):
        tag_ids = tagid_batches[batch_num]
        seq_len_list = seq_len_batches[batch_num]
        with tf.GradientTape() as tape:
            logits = mymodel(seq_ids)
            loss = mymodel.crf_loss(logits, tag_ids, seq_len_list)
            pred_tags, pred_best_score = crf.crf_decode(potentials=logits, transition_params=mymodel.trans_p,
                                                        sequence_length=seq_len_list)
        # p_tags = findall_tag(pred_tags)
        # t_tags = findall_tag(tag_ids)
        # (P, R, F1) = P_R_F1_score(p_tags, t_tags)
        print('epoch:{}\t\tbatch:{}\t\tloss:{:.2f}\t\tP :{:.8f}\t\tR :{:.8f}\t\tF1 :{:.8f}\t\t'.format(epoch_num,batch_num, loss, 0, 0, 0))
        grads = tape.gradient(loss, mymodel.trainable_variables)
        optimizer.apply_gradients(zip(grads, mymodel.trainable_variables))
        # optimizer.minimize(loss, [myModel_bilstm.trainable_variables])
        if batch_num % 20 == 0:
            manager.save(checkpoint_number=batch_num + epoch_num * len(charid_batches))


if __name__ == '__main__':
    from DataUtils2 import read_train_data, get_batches

    starttime = time.time()
    train_data_path = r'F:\zzd\毕业论文\论文代码\NER\data\someNEWS_BIOES.dev'
    vocab_path = r'F:\zzd\毕业论文\论文代码\NER\vocab\vocab.pkl'
    data = read_train_data(train_data_path, vocab_path)
    (charid_batches, tagid_batches, seq_len_batches) = get_batches(data, 200)

    endtime = time.time()
    print('batches is ready! cost time:',endtime-starttime)
    starttime= time.time()

    configers = conf()
    model = Model_bilstm(configers)
    for epoch in range(100):
        train_one_epoch(model, charid_batches, tagid_batches, seq_len_batches, epoch_num=epoch)

    endtime = time.time()
    print('done!——————run time ：', str(endtime - starttime), 's.')
    #     starttime = time.time()
    # tf.test.is_gpu_available()
    # train_data_path = r'F:\zzd\毕业论文\论文代码\NER\data\someNEWS_BIOES.dev'
    # vocab_path = r'F:\zzd\毕业论文\论文代码\NER\vocab\vocab.pkl'
    # data = read_train_data(train_data_path, vocab_path)
    # batches = get_batches(data, 100)
    # batch = batches[0]
    # seq_ids, tag_ids, seq_len_list = get_train_data_from_batch(batch)
    # tagdict = findall_tag(tag_ids)
    # print(tagdict)
