import datetime
from MyModel_V4 import Model_NER, conf
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

mod = 'BiLSTM'
batch_size = 100
recordFileName = '_'.join(
    ['3L_BertBiLSTM',  str(batch_size) + 'bs','pre+ft','movie-scene-government'])
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
log_dir_vali = 'Records/' + recordFileName + '/tensorboard/' + '-vali'
log_dir_test_train = 'Records/' + recordFileName + '/tensorboard/' + '-test_train'
log_dir_test_vali = 'Records/' + recordFileName + '/tensorboard/' + '-test_vali'
log_writer_train = tf.summary.create_file_writer(log_dir_train)
log_writer_vali = tf.summary.create_file_writer(log_dir_vali)
log_writer_vali_train = tf.summary.create_file_writer(log_dir_test_train)
log_writer_vali_test = tf.summary.create_file_writer(log_dir_test_vali)
#-------------------------------------------------------

print('record files is created!\n'
      '-----------------------------------------------------------\n')

train_tasks = ['address', 'game', 'organization',
               'company', 'name', 'book','position']
test_tasks = ['government', 'scene', 'movie']


# 获取预训练的训练集数据： 分两部分，预训练_训练集 和 预训练_测试集
#-------------------------------------------------------
train_batches = []
pre_train_data_path = os.path.join('data', 'CLUE_BIOES_train')
pre_vali_data_path = os.path.join('data', 'CLUE_BIOES_dev')
pre_train_batches = get_batches_v1(train_data_path=pre_train_data_path, batchsize=200, taskname=train_tasks)
pre_vali_batches = get_batches_v1(train_data_path=pre_vali_data_path, batchsize=200, taskname=train_tasks)

pre_train_batch_pred = bert_embedding(pre_train_batches)
pre_vali_batches_pred = bert_embedding(pre_vali_batches)


# 获取测试集数据： 测试集分两部分，测试_训练集 和 测试_测试集
#-------------------------------------------------------
test_train_data_paths = []
for t in test_tasks:
    temp = os.path.join('data_tasks', t)
    test_train_data_paths.append(temp)
test_train_batches = get_batches_v4(train_data_path_list=test_train_data_paths, batch_size=batch_size, batch_num=1,
                                    taskname=test_tasks)
test_vali_data_path = ['data/CLUE_BIOES_dev']
test_vali_batches = get_batches_v4(train_data_path_list=test_vali_data_path, batch_size=500, batch_num=1,
                                 taskname=test_tasks)

test_train_batches_pred = bert_embedding(test_train_batches)
test_vali_batch_pred = bert_embedding(test_vali_batches)
#-------------------------------------------------------

# 指数衰减学习率
exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.005, decay_steps=1, decay_rate=0.96)
myModel.optimizer = tf.optimizers.Adam(exponential_decay)

# 预训练开始
#-------------------------------------------------------
max_F1 = 0
for epoch in range(epochNum, 500):
    starttime = time.time()
    loss_t,P_t,R_T,F1_t = myModel.inner_train_one_step(pre_train_batch_pred, inner_iters=0, inner_epochNum=epoch,
                                                       outer_epochNum=0,
                                                       task_name=train_tasks, log_writer=log_writer_train,mod='pretrain')
    myModel.save_weights(ckpt_path_theta_0)
    print('train loss:{}, train F1:{} <-----------------\n'.format(loss_t,F1_t))
    print('**********************************************\n')
    # 在测试集上看结果
    test_loss, pred_tags_masked, tag_ids_padded,P, R, F1 = myModel.validate_one_batches(pre_vali_batches, train_tasks,
                                                                                        log_writer_vali, epoch)

    if F1 > max_F1:
        max_F1 = F1
        myModel.save_weights(ckpt_path_theta_t)

    # 记录epoch
    Record_epoch_num(recordFileName, epoch)

    endtime = time.time()
    print('done inner epoch:{}!——————run time ：{}s**********************************************.\n'.format(epoch, str(endtime - starttime)))

# 微调阶段
#-------------------------------------------------------
max_F1 = 0
for epoch in range(100):
    loss_t, P_t, R_T, F1_t = myModel.inner_train_one_step(test_train_batches_pred, inner_iters=0, inner_epochNum=epoch,
                                                          outer_epochNum=0,
                                                          task_name=test_tasks, log_writer=log_writer_vali_train,mod='finetune')
    myModel.save_weights(ckpt_path_theta_0)
    print('train loss:{}, train F1:{} <-----------------\n'.format(loss_t, F1_t))
    print('**********************************************\n')
    # 在测试集上看结果
    test_loss, pred_tags_masked, tag_ids_padded, P, R, F1 = myModel.validate_one_batches(test_vali_batch_pred, test_tasks,
                                                                                         log_writer_vali_test, epoch)
    if F1 > max_F1:
        max_F1 = F1
        myModel.save_weights(ckpt_path_theta_t)
        content = 'P\t{}\nR\t{}\nF1\t{}\n'.format(P,R,max_F1)
        with open(maxPRF1filepath,'w',encoding='utf-8') as f:
            f.write(content)