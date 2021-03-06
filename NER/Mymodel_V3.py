import datetime
import os

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow_addons.text import crf
from Utils import *
import time
from tqdm import tqdm
from conlleval import evaluate

tf.config.set_soft_device_placement(True)
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class conf():
    def __init__(self,choose_mod=None):
        self.LSTM_dim = 300
        self.tag_num = 18
        self.optimizer = tf.optimizers.Adam(learning_rate=0.005)
        self.batchsize = 200
        self.embedding_in_dim = 10000
        self.embedding_out_dim = 100
        if choose_mod == None:
            self.mod = 'BiLSTM'
        else:
            self.mod = choose_mod




class Model_NER(keras.Model):
    def __init__(self, conf:conf):
        super(Model_NER, self).__init__()
        self.mod = conf.mod

        # 模型基本参数
        self.LSTM_dim = conf.LSTM_dim
        self.tag_num = conf.tag_num
        # 模型所需的层定义
        self.embedding = layers.Embedding(input_dim=conf.embedding_in_dim, output_dim=conf.embedding_out_dim, mask_zero=True)

        self.dense = layers.Dense(self.tag_num)
        self.fw_LSTM = layers.LSTM(units=self.LSTM_dim, return_sequences=True, go_backwards=False)
        self.bw_LSTM = layers.LSTM(units=self.LSTM_dim, return_sequences=True, go_backwards=True)
        self.BiLSTM = layers.Bidirectional(self.fw_LSTM, backward_layer=self.bw_LSTM)
        self.trans_p = tf.Variable(
            tf.keras.initializers.GlorotUniform()([self.tag_num, self.tag_num]), name="transition_matrix"
        )

        self.conv1 = layers.Conv1D(conf.embedding_out_dim,3,padding='same',dilation_rate=1,activation='relu')
        self.conv2 = layers.Conv1D(conf.embedding_out_dim,3,padding='same',dilation_rate=1,activation='relu')
        self.conv3 = layers.Conv1D(conf.embedding_out_dim,3,padding='same',dilation_rate=2,activation='relu')

        self.optimizer = conf.optimizer

    def call(self, inputs,training=None, mask=None):
        x = self.embedding(inputs)
        if self.mod == 'IDCNN':
            conc_x = []
            for i in range(4):
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                conc_x.append(x)
            x = tf.concat(conc_x,axis=2)
            x = self.dense(x)
        else:
            x = self.BiLSTM(x)
            x = self.dense(x)
        return x

    def crf_loss(self, input, tag_ids, sentence_len_list):
        likehood, self.trans_p = crf.crf_log_likelihood(inputs=input, tag_indices=tag_ids,
                                                        transition_params=self.trans_p,
                                                        sequence_lengths=sentence_len_list)
        loss = tf.reduce_mean(-likehood)
        return loss

    def predict_one_batch(self, test_batch):
        seq_ids_padded, tag_ids_padded, seq_len_list = get_train_data_from_batch(test_batch)
        logits = self(seq_ids_padded)
        loss = self.crf_loss(logits, tag_ids_padded, seq_len_list)
        pred_tags, pred_best_score = crf.crf_decode(potentials=logits, transition_params=self.trans_p,
                                                    sequence_length=seq_len_list)
        pred_tags_masked = seq_masking(pred_tags, seq_len_list)
        return (loss, pred_tags_masked, tag_ids_padded)

    def inner_train_one_step(self,batches, inner_iters, inner_epochNum, outer_epochNum, task_name,
                             log_writer):
        '''
        :param self:
        :param batches: one batch data: [[sentence],[sentence],....]
                               sentence=[[chars],[charids],[tags],[tag_ids]]
        :param inner_epochNum:
        :return:
        '''

        batches_len = len(batches)

        # =====run model=======
        for batch_num in range(batches_len):
            batch = batches[batch_num]
            seq_ids_padded, tag_ids_padded, seq_len_list = get_train_data_from_batch(batch)
            with tf.GradientTape() as tape:
                logits = self(seq_ids_padded)
                loss = self.crf_loss(logits, tag_ids_padded, seq_len_list)
                pred_tags, pred_best_score = crf.crf_decode(potentials=logits, transition_params=self.trans_p,
                                                            sequence_length=seq_len_list)
            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            # optimizer.minimize(loss, [myModel_bilstm.trainable_variables])

        pred_tags_masked = seq_masking(pred_tags, seq_len_list)
        p_tags_char, p_tagsid_flatten = get_id2tag(pred_tags_masked, taskname=task_name)
        t_tags_char, t_tagsid_flatten = get_id2tag(tag_ids_padded, taskname=task_name)
        (P_t, R_t, F1_t), _ = evaluate(t_tags_char, p_tags_char, verbose=False)
        with log_writer.as_default():
            step = batch_num + 1 + inner_epochNum * batches_len
            tf.summary.scalar("loss", loss, step=inner_epochNum + outer_epochNum * inner_iters)
            tf.summary.scalar("P", P_t, step=inner_epochNum)
            tf.summary.scalar("R", R_t, step=inner_epochNum)
            tf.summary.scalar("F", F1_t, step=inner_epochNum)
        return (loss,P_t)

# if __name__ == '__main__':
