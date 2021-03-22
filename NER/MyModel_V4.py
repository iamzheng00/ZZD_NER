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

        self.tag_num = 16
        self.optimizer = tf.optimizers.Adam(learning_rate=0.005)
        self.batchsize = 200

        if choose_mod == None:
            self.mod = 'BiLSTM'
        else:
            self.mod = choose_mod




class Model_NER(keras.Model):
    """
    没有embedding层，需做额外的embedding 再输入
    """
    def __init__(self, conf:conf):
        super(Model_NER, self).__init__()
        self.mod = conf.mod

        # 模型基本参数
        self.tag_num = conf.tag_num
        # 模型所需的层定义

        self.LSTM1 = layers.LSTM(300, return_sequences=True, go_backwards=False)
        self.LSTM2 = layers.LSTM(300, return_sequences=True, go_backwards=True)
        self.BiLSTM = layers.Bidirectional(self.LSTM1, backward_layer=self.LSTM2)
        self.trans_p = tf.Variable(
            tf.keras.initializers.GlorotUniform()([self.tag_num, self.tag_num]), name="transition_matrix"
        )
        self.dense = layers.Dense(self.tag_num)
        self.optimizer = conf.optimizer

    def call(self, inputs,training=None, mask=None):
        x = self.BiLSTM(inputs)
        x = self.dense(x)
        return x

    def crf_loss(self, input, tag_ids, sentence_len_list):
        likehood, self.trans_p = crf.crf_log_likelihood(inputs=input, tag_indices=tag_ids,
                                                        transition_params=self.trans_p,
                                                        sequence_lengths=sentence_len_list)
        loss = tf.reduce_mean(-likehood)
        return loss

    def validate_one_batch(self, test_batch, task_name, log_writer, epoch):
        seq_embeddings = test_batch['emb']
        tag_ids = test_batch['tag_ids']
        seq_len_list = test_batch['lens']
        seq_len_list_plus2 = [x + 2 for x in seq_len_list]
        tag_ids_padded = pad_tag_ids(tag_ids)

        logits = self(seq_embeddings)
        loss = self.crf_loss(logits, tag_ids_padded, seq_len_list_plus2)
        pred_tags, pred_best_score = crf.crf_decode(potentials=logits, transition_params=self.trans_p,
                                                    sequence_length=seq_len_list_plus2)
        pred_tags_masked = seq_masking(pred_tags, seq_len_list_plus2)
        p_tags_char, _ = get_id2tag_V2(pred_tags_masked, seq_len_list_plus2, taskname=task_name)
        t_tags_char, _ = get_id2tag_V2(tag_ids_padded, seq_len_list_plus2, taskname=task_name)
        (P, R, F1), _ = evaluate(t_tags_char, p_tags_char, verbose=True)
        write_to_log(loss, P, R, F1, t_tags_char, log_writer, epoch)
        return (loss, pred_tags_masked, tag_ids_padded,P, R, F1)

    def inner_train_one_step(self,batches, inner_iters, inner_epochNum, outer_epochNum, task_name,
                             log_writer):
        '''
        :param self:
        :param batches: one batch data: [[sentence],[sentence],....]
                               sentence=[emb:[],chars:[],tags:[],tag_ids:[]]
        :param inner_epochNum:
        :return:
        '''

        batches_len = len(batches)

        # =====run model=======
        for batch_num in range(batches_len):
            batch = batches[batch_num]
            seq_embeddings= batch['emb']
            tag_ids= batch['tag_ids']
            seq_len_list = batch['lens']
            seq_len_list_plus2 = [x+2 for x in seq_len_list]
            tag_ids_padded = pad_tag_ids(tag_ids)

            with tf.GradientTape() as tape:
                logits = self(seq_embeddings)
                loss = self.crf_loss(logits, tag_ids_padded, seq_len_list_plus2)
                pred_tags, pred_best_score = crf.crf_decode(potentials=logits, transition_params=self.trans_p,
                                                            sequence_length=seq_len_list_plus2)
            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            # optimizer.minimize(loss, [myModel_bilstm.trainable_variables])

        pred_tags_masked = seq_masking(pred_tags, seq_len_list_plus2)
        p_tags_char, p_tagsid_flatten = get_id2tag_V2(pred_tags_masked, seq_len_list_plus2, taskname=task_name)
        t_tags_char, t_tagsid_flatten = get_id2tag_V2(tag_ids_padded, seq_len_list_plus2, taskname=task_name)
        (P_t, R_t, F1_t), _ = evaluate(t_tags_char, p_tags_char, verbose=False)
        with log_writer.as_default():
            # step = batch_num + 1 + inner_epochNum * batches_len
            tf.summary.scalar("loss", loss, step=inner_epochNum + outer_epochNum * inner_iters)
            tf.summary.scalar("P", P_t, step=inner_epochNum)
            tf.summary.scalar("R", R_t, step=inner_epochNum)
            tf.summary.scalar("F", F1_t, step=inner_epochNum)
        return (loss,F1_t)

if __name__ == '__main__':

    cof = conf()
    model = Model_NER(cof)
    e = tf.ones([200,52,768])
    x = model(e)
    print(model.BiLSTM.trainable_variables)
