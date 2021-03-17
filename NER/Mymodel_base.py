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
    def __init__(self, choose_mod=None):
        self.LSTM_dim = 300
        self.tag_num = 42
        self.optimizer = tf.optimizers.Adam(learning_rate=0.001)
        self.batchsize = 200
        if choose_mod == None:
            self.mod = 'BiLSTM'
        else:
            self.mod = choose_mod


class Model_NER(keras.Model):
    def __init__(self, conf):
        super(Model_NER, self).__init__()
        self.mod = conf.mod

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

        self.conv1 = layers.Conv1D(100, 3, padding='same', dilation_rate=1, activation='relu')
        self.conv2 = layers.Conv1D(100, 3, padding='same', dilation_rate=1, activation='relu')
        self.conv3 = layers.Conv1D(100, 3, padding='same', dilation_rate=2, activation='relu')

        self.optimizer = conf.optimizer

    def call(self, inputs, training=None, mask=None):
        x = self.embedding(inputs)
        if self.mod == 'IDCNN':
            conc_x = []
            for i in range(4):
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                conc_x.append(x)
            x = tf.concat(values=conc_x, axis=2)
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

    def inner_train_one_step(self, batches, epochNum, task_name, log_writer,log_dir):
        '''
        :param self:
        :param batches: one batch data: [[sentence],[sentence],....]
                               sentence=[[chars],[charids],[tags],[tag_ids]]
        :param inner_epochNum:
        :return:
        '''
        # tf.summary.trace_on(graph=True,profiler=True)  # 开启Trace（可选）
        batch_Nums = len(batches)

        losses,P_ts,R_ts,F1_ts = [],[],[],[]
        # =====run model=======
        with tqdm(total=batch_Nums) as bar:
            for batch_num in range(batch_Nums):
                batch = batches[batch_num]
                seq_ids_padded, tag_ids_padded, seq_len_list = get_train_data_from_batch(batch)
                with tf.GradientTape() as tape:
                    # print(batch[0]) # 调试用
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
                losses.append(loss)
                P_ts.append(P_t)
                R_ts.append(R_t)
                F1_ts.append(F1_t)
                print('train_loss:{}, train_P:{}'.format(loss,P_t))
                bar.update(1)
        with log_writer.as_default():
            tf.summary.scalar("loss", np.mean(losses), step=epochNum)
            tf.summary.scalar("P", np.mean(P_ts), step=epochNum)
            tf.summary.scalar("R", np.mean(R_ts), step=epochNum)
            tf.summary.scalar("F1", np.mean(F1_ts), step=epochNum)
            # tf.summary.trace_export(name="model_trace", step=epochNum, profiler_outdir=log_dir)    # 保存Trace信息到文件
# if __name__ == '__main__':
