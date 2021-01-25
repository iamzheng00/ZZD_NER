# encoding=utf-8
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow_addons.text import crf


class conf():
    LSTM_dim=100
    tag_num=100

class myModel_bilstm(keras.Model):
    def __init__(self, conf, ):
        super(myModel_bilstm, self).__init__()
        # 模型基本参数
        self.LSTM_dim = conf.LSTM_dim
        self.tag_num = conf.tag_num
        # 模型所需的层定义
        self.embedding = layers.Embedding(10000,100, mask_zero=True)
        self.dense = layers.Dense(self.tag_num)
        self.fw_LSTM = layers.LSTM(units=self.LSTM_dim, return_sequences=True, go_backwards=False)
        self.bw_LSTM = layers.LSTM(units=self.LSTM_dim, return_sequences=True, go_backwards=True)
        BiLSTM = layers.Bidirectional(self.fw_LSTM, backward_layer=self.bw_LSTM)

    def call(self, inputs, training=None, mask=None):
        x = self.embedding(inputs)
        x = self.BiLSTM(x)
        x = self.dense(x)

        return x

    def embedding_layer(self):
        pass # todo

    def crf_loss(self,input,tag_ids,sentence_len):
        loss, self.trans_p = crf.crf_log_likelihood(inputs=input, tag_indices=tag_ids,sequence_lengths=sentence_len)
        return tf.reduce_sum(loss)


def train_one_epoch(model, batches,ckp_path,epoch_num):
    configers = conf()
    optimizer = tf.optimizers.Adam(learning_rate=.0005)
    myBiLSTM = myModel_bilstm(configers)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=myBiLSTM)
    manager = tf.train.CheckpointManager(checkpoint, directory='checkpoints', max_to_keep=5)
    # batches process todo

    # =====run model=======
    for batch_num, (seqs, tag_ids) in enumerate(batches):
        seq_len = maxlen(seqs)
        with tf.GradientTape(persistent=True) as tape:
            logits = myBiLSTM(seqs)
            loss=myBiLSTM.crf_loss(seqs,tag_ids,seq_len)
            seq_tags, best_score = crf.crf_decode(potentials=logits,transition_params=myBiLSTM.trans_p, sequence_length=seq_len)
        grads = tape.gradient(loss, myBiLSTM.trainable_variables)
        optimizer.apply_gradients(zip(grads,myBiLSTM.trainable_variables))
        # optimizer.minimize(loss, [myModel_bilstm.trainable_variables])
        if batch_num%50==0:
            manager.save(checkpoint_number=batch_num+epoch_num*len(batches))



