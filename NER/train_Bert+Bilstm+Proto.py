import datetime
from MyModel_V5proto import Model_NER, conf
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow_addons.text import crf
from Utils import *
import time
from tqdm import tqdm
from conlleval import evaluate
from transformers import BertTokenizer, TFBertModel


tokenizer_savepath = 'bert/tokenizer'
bertmodel_savepath = 'bert/model'

tokenizer = BertTokenizer.from_pretrained(tokenizer_savepath)
bertmodel = TFBertModel.from_pretrained(bertmodel_savepath)

def bert_embedding(batches):

    batches_prd = []
    for ba in batches:
        '''
        ba=[sentence,sentence...]
        sentence = [[chars],[tags],[tag_ids]]
        '''
        chars = [x[0] for x in ba]
        tags = [x[1] for x in ba]
        tag_ids = [x[2] for x in ba]
        lens = [len(x[0]) for x in ba]
        ml = max(lens)
        chars_ids = []
        for sent in chars:
            tokenized = tokenizer.encode(sent, max_length=ml + 2, padding='max_length', return_tensors='tf')
            chars_ids.append(tokenized)
        bert_input = tf.squeeze(tf.stack(chars_ids, axis=1))
        emb = bertmodel(bert_input).last_hidden_state
        batches_prd.append({
            'emb': emb,
            'chars': chars,
            'tag_ids': tag_ids,
            'tags': tags,
            'lens': lens
        })
    return batches_prd


tf.config.set_soft_device_placement(True)
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

mod = 'BiLSTM+proto'
batch_size = 2
recordFileName = '_'.join(
    ['3L_Bert' + mod,  str(batch_size) + 'bs'])
create_record_dirs(recordFileName)
epochNum = get_epochNum(recordFileName)  # 获取当前记录的epoch数

# 配置模型参数、检查点
#-------------------------------------------------------
configers = conf(choose_mod=mod)
myModel = Model_NER(configers)
ckpt_dir_inner = os.path.join('Records',recordFileName, 'checkpoints')
ckpt_dir_theta_0 = os.path.join('Records',recordFileName, 'theta_0') # 存 每一轮外层循环后学到的初始参数
ckpt_dir_theta_t = os.path.join('Records',recordFileName, 'theta_t') # 存 表现最佳的初始参数
ckpt_dir_vali_theta = os.path.join('Records',recordFileName, 'theta_vali') # 存 在验证集测试时，模型训练过程中的参数

ckpt_path_theta_0 = os.path.join(ckpt_dir_theta_0, 'ckpt_theta_0')
ckpt_path_theta_t = os.path.join(ckpt_dir_theta_t, 'ckpt_theta_t')
ckpt_path_vali_theta = os.path.join(ckpt_dir_vali_theta, 'ckpt_theta_vali_train')
maxPRF1filepath = os.path.join('Records',recordFileName,'mPRF1.txt')
checkpoint = tf.train.Checkpoint(optimizer=myModel.optimizer, model=myModel)
ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=ckpt_dir_inner, max_to_keep=5)
#-------------------------------------------------------
if epochNum == 0:
    myModel.save_weights(ckpt_path_theta_0)
else:
    myModel.load_weights(ckpt_path_theta_0)

# 配置tensorboard
#-------------------------------------------------------

log_dir_train = 'Records/' + recordFileName + '/tensorboard/' + '-train'
log_dir_vali_train = 'Records/' + recordFileName + '/tensorboard/' + '-vali_train'
log_dir_vali_test = 'Records/' + recordFileName + '/tensorboard/' + '-vali_test'
log_writer_train = tf.summary.create_file_writer(log_dir_train)
log_writer_vali_train = tf.summary.create_file_writer(log_dir_vali_train)
log_writer_vali_test = tf.summary.create_file_writer(log_dir_vali_test)
#-------------------------------------------------------

print('record files is created!\n'
      '-----------------------------------------------------------\n')

train_tasks = ['address', 'scene', 'government', 'organization',
               'company', 'name', 'book']
tasks = ['game', 'position', 'movie']
# 获取数据集路径：
#-------------------------------------------------------
train_data_path = []
for t in tasks:
    temp = os.path.join('data_tasks', t)
    train_data_path.append(temp)
# vali_train_batch = get_batches_FS(train_data_path_list=vali_train_data_paths, S_size=50, Q_size=50,
#                                         taskname=validation_tasks)
# vali_test_data_path = ['data/CLUE_BIOES_dev']
# vali_test_batch = get_batches_FS(train_data_path_list=vali_test_data_path, S_size=50, Q_size=200,
#                                  taskname=validation_tasks)
#
# vali_test_batch_pred = bert_embedding([vali_test_batch])[0]
# vali_train_batches_pred = bert_embedding(vali_train_batches)
#-------------------------------------------------------

# 指数衰减学习率
# exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=0.05, decay_steps=50, decay_rate=0.8)
# myModel.optimizer = tf.optimizers.Adam(exponential_decay)
# myModel.optimizer = tf.optimizers.Adam(learning_rate=0.001)

max_F1 = 0
# 训练开始
#-------------------------------------------------------
for epoch in range(epochNum, epochNum+500):
    # starttime = time.time()
    print('epoch:{}'.format(epoch))
    # 任务采样
    train_set = get_batch_FS(train_data_path_list=train_data_path, S_size=50, Q_size=50,
                             taskname=tasks)
    # 获得字嵌入
    train_batch_embedded = bert_embedding(train_set)
    loss_t,P_t,R_T,F1_t = myModel.inner_train_one_step(train_batch_embedded, inner_iters=0, inner_epochNum=epoch,
                                                       outer_epochNum=0,
                                                       task_name=tasks, log_writer=log_writer_vali_train)
    myModel.save_weights(ckpt_path_theta_0)
    print('train loss:{}, train F1:{} <-----------------\n'.format(loss_t,F1_t))
    print('**********************************************\n')

    if epoch%10==0:
        # 在测试集上看结果
        test_data_path = ['data/CLUE_BIOES_dev']
        test_set = get_batch_FS(train_data_path_list=test_data_path, S_size=50, Q_size=200,
                                taskname=tasks)
        test_batch_embedded = bert_embedding(test_set)
        test_loss, pred_tags_masked, tag_ids_padded,P, R, F1 = myModel.validate_one_batch(test_batch_embedded, tasks,
                                                                                          log_writer_vali_test, epoch)
        if F1 > max_F1:
            max_F1 = F1
            myModel.save_weights(ckpt_path_theta_t)
            content = 'P\t{}\nR\t{}\nF1\t{}\n'.format(P,R,max_F1)
            with open(maxPRF1filepath,'w',encoding='utf-8') as f:
                f.write(content)
    # 记录epoch
    Record_epoch_num(recordFileName, epoch)

    # endtime = time.time()
    # print('done inner epoch:{}!——————run time ：{}s**********************************************.\n'.format(epoch, str(endtime - starttime)))
