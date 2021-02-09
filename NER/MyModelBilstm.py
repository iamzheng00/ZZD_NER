import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow_addons.text import crf
from DataUtils import maxlen, findall_tag, P_R_F1_score
import time
from tqdm import tqdm


class conf():
    LSTM_dim = 300
    tag_num = 18


class Model_bilstm(keras.Model):
    def __init__(self, conf, ):
        super(Model_bilstm, self).__init__()
        # 模型基本参数
        self.LSTM_dim = conf.LSTM_dim
        self.tag_num = conf.tag_num
        # 模型所需的层定义
        self.embedding = layers.Embedding(input_dim=10000, output_dim=100, mask_zero=True)

        self.dense = layers.Dense(self.tag_num)
        self.fw_LSTM = layers.LSTM(units=self.LSTM_dim, return_sequences=True, go_backwards=False)
        self.bw_LSTM = layers.LSTM(units=self.LSTM_dim, return_sequences=True, go_backwards=True)
        self.BiLSTM = layers.Bidirectional(self.fw_LSTM, backward_layer=self.bw_LSTM)
        self.trans_p = tf.Variable(
            tf.keras.initializers.GlorotUniform()([self.tag_num, self.tag_num]), name="transition_matrix"
        )

    def call(self, inputs, training=None, mask=None):
        x = self.embedding(inputs)
        x = self.BiLSTM(x)
        x = self.dense(x)
        return x

    def crf_loss(self, input, tag_ids, sentence_len_list):
        likehood, self.trans_p = crf.crf_log_likelihood(inputs=input, tag_indices=tag_ids,
                                                        transition_params=self.trans_p,
                                                        sequence_lengths=sentence_len_list)
        loss = tf.reduce_mean(-likehood)
        return loss


# 从批数据中 分离出seq_ids 和 tag_ids,以及每一句的长度列表：seq_len_list
def get_train_data_from_batch(batch):
    '''
    :param batch: [[sentence],[sentence],....]
            sentence=[[chars],[charids],[tags],[tag_ids]]
    :return:
    '''
    seq_ids, tag_ids = [], []
    for sentence in batch:
        seq_ids.append(sentence[1])
        tag_ids.append(sentence[3])
    seq_len_list = [(len(s)+1) for s in seq_ids]
    seq_ids = keras.preprocessing.sequence.pad_sequences(seq_ids,maxlen=max(seq_len_list), padding='post', value=0)
    tag_ids = keras.preprocessing.sequence.pad_sequences(tag_ids,maxlen=max(seq_len_list), padding='post', value=0)
    seq_ids = tf.convert_to_tensor(seq_ids, dtype='int32')
    tag_ids = tf.convert_to_tensor(tag_ids, dtype='int32')
    return (seq_ids, tag_ids, seq_len_list)


def seq_masking(seqs, lens):
    mask = tf.sequence_mask(lengths=lens, dtype=tf.int32)
    seqs_masked = seqs * mask
    return seqs_masked


def train_one_epoch(mymodel, batches, epoch_num=1, checkpoints_dir='checkpoints00'):
    '''
    :param mymodel:
    :param batches: one batch data: [[sentence],[sentence],....]
                           sentence=[[chars],[charids],[tags],[tag_ids]]
    :param epoch_num:
    :return:
    '''
    optimizer = tf.optimizers.Adam(learning_rate=.005)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=mymodel)
    manager = tf.train.CheckpointManager(checkpoint, directory=checkpoints_dir, max_to_keep=10)

    # =====run model=======
    for batch_num in tqdm(range(len(batches))):
        batch = batches[batch_num]
        seq_ids_padded, tag_ids_padded, seq_len_list = get_train_data_from_batch(batch)
        with tf.GradientTape() as tape:
            logits = mymodel(seq_ids_padded)
            loss = mymodel.crf_loss(logits, tag_ids_padded, seq_len_list)
            pred_tags, pred_best_score = crf.crf_decode(potentials=logits, transition_params=mymodel.trans_p,
                                                        sequence_length=seq_len_list)
        pred_tags_masked = seq_masking(pred_tags, seq_len_list)
        p_tags = findall_tag(pred_tags_masked, seq_len_list)
        t_tags = findall_tag(tag_ids_padded, seq_len_list)
        (P, R, F1) = P_R_F1_score(p_tags, t_tags)

        grads = tape.gradient(loss, mymodel.trainable_variables)
        optimizer.apply_gradients(zip(grads, mymodel.trainable_variables))
        # optimizer.minimize(loss, [myModel_bilstm.trainable_variables])
        if batch_num % 20 == 0:
            print(
                'epoch:{}\t\tbatch:{}\t\ttrain_loss:{:.2f}\t\ttrain_P :{:.8f}\t\ttrain_R :{:.8f}\t\ttrain_F1 :{:.8f}\t\t'.format(
                    epoch_num,
                    batch_num,
                    loss,
                    P, R, F1))
    manager.save(checkpoint_number=epoch_num)


if __name__ == '__main__':
    from DataUtils import read_train_data, get_batches

    starttime = time.time()
    train_data_path = r'F:\zzd\毕业论文\论文代码\NER\data\someNEWS_BIOES.dev'
    vocab_path = r'F:\zzd\毕业论文\论文代码\NER\vocab\vocab.pkl'
    data = read_train_data(train_data_path, vocab_path)
    batches = get_batches(data, 100)

    # seq_ids, tag_ids, seq_len_list = get_train_data_from_batch(batches[1])
    # print('done')
    # batches = tf.convert_to_tensor(batches,dtype='int32')

    endtime = time.time()
    print('batches is ready! cost time:', endtime - starttime)
    starttime = time.time()

    configers = conf()
    model = Model_bilstm(configers)
    ckpt_dir = 'checkpoints02'
    for epoch in range(10):
        train_one_epoch(model, batches, epoch_num=epoch, checkpoints_dir=ckpt_dir)

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
