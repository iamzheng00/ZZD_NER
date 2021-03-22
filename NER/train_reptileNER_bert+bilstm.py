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

# lstm1 = layers.LSTM(300, return_sequences=True, go_backwards=False)
# lstm2 = layers.LSTM(300, return_sequences=True, go_backwards=True)
# bilstm = layers.Bidirectional(lstm1, backward_layer=lstm2)
# dense = layers.Dense(3)

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bertmodel = TFBertModel.from_pretrained('bert-base-chinese')


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
inner_iters = 8
e = 0.1
e_final = 0.001
batch_size = 100
recordFileName = '_'.join(
    ['record_3L_reptile+Bert' + mod, str(inner_iters) + 'i', str(e) + '-' + str(e_final) + 'e', str(batch_size) + 'bs'])
create_record_dirs(recordFileName)
epochNum = get_epochNum(recordFileName)  # 获取当前记录的epoch数

# 配置模型参数、检查点
configers = conf(choose_mod=mod)
myModel = Model_NER(configers)
ckpt_dir_inner = os.path.join(recordFileName, 'checkpoints')
ckpt_dir_theta_0 = os.path.join(recordFileName, 'theta_0')
ckpt_path_theta_0 = os.path.join(ckpt_dir_theta_0, 'ckpt_theta_0')
ckpt_dir_theta_t = os.path.join(recordFileName, 'theta_t') # 用来存 在验证集上训练模型的最佳参数
# ckpt_path_theta_t = os.path.join(ckpt_dir_theta_t, 'ckpt_theta_t')
checkpoint = tf.train.Checkpoint(optimizer=myModel.optimizer, model=myModel)
ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=ckpt_dir_inner, max_to_keep=5)

if epochNum == 0:
    myModel.save_weights(ckpt_path_theta_0)
else:
    myModel.load_weights(ckpt_path_theta_0)

# 配置tensorboard
# log_dir_train = recordFileName + '/tensorboard/' + datetime.datetime.now().strftime("%Y%m%d-%H%M") + '-train'
# log_dir_test = recordFileName + '/tensorboard/' + datetime.datetime.now().strftime("%Y%m%d-%H%M") + '-test'
log_dir_train = recordFileName + '/tensorboard/' + '-train'
log_dir_vali_train = recordFileName + '/tensorboard/' + '-vali_train'
log_dir_vali_test = recordFileName + '/tensorboard/' + '-vali_test'
log_writer_train = tf.summary.create_file_writer(log_dir_train)
log_writer_vali_train = tf.summary.create_file_writer(log_dir_vali_train)
log_writer_vali_test = tf.summary.create_file_writer(log_dir_vali_test)

print('record files is created!\n'
      '-----------------------------------------------------------\n')

train_tasks = ['address', 'scene', 'government', 'organization',
               'company', 'name', 'book']
validation_tasks = ['game', 'position', 'movie']
# 获取验证集数据： 验证集分两部分，验证_训练集 和 验证_测试集
vali_train_data_paths = []
for t in validation_tasks:
    temp = os.path.join('data_tasks', t)
    vali_train_data_paths.append(temp)

vali_test_data_path = ['data/CLUE_BIOES_dev']
vali_test_batch = get_batches_v4(train_data_path_list=vali_test_data_path, batch_size=10000, batch_num=1,
                                 taskname=validation_tasks)

vali_test_batch_pred = bert_embedding([vali_test_batch])[0]

for epoch in range(epochNum, 1000):
    meta_step_size = epoch / 1000 * e_final + (1 - epoch / 1000) * e
    starttime = time.time()
    print('====outer epoch:{}==========================================='.format(epoch))
    vars_list = []
    myModel.load_weights(ckpt_path_theta_0)
    # ==========对每个任务进行训练=============
    for task_N in range(5):
        # 任务采样，获取训练数据文件地址
        task_samples = random.sample(train_tasks, 3)
        task_samples.sort()

        path_list = []
        for task in task_samples:
            p = os.path.join('data_tasks', task)
            path_list.append(p)

        print('task{}:{},{},{}================'.format(task_N, task_samples[0], task_samples[1], task_samples[2]))
        # NER模型载入参数
        # if epoch == 0:
        #     myModel.load_weights(ckpt_path_theta_0)
        # else:
        #     myModel.load_weights(ckpt_path_theta_t)
        myModel.optimizer = tf.optimizers.Adam(learning_rate=0.002)

        # ==============内循环训练阶段===================
        batches = get_batches_v4(train_data_path_list=path_list, batch_size=batch_size, batch_num=1,
                                 taskname=task_samples)
        batches_prd = bert_embedding(batches)
        with tqdm(total=inner_iters) as bar:
            for i in range(inner_iters):
                train_loss, train_P = myModel.inner_train_one_step(batches_prd, inner_iters, i, epoch,
                                                                   task_name=task_samples, log_writer=log_writer_train)
                bar.update(1)
            print('train_loss:{}   train_Precision:{}'.format(train_loss, train_P))

        # 记录当前任务训练所得model的参数
        task_names = '-'.join(task_samples)
        # myModel.save_weights(ckpt_dir_inner + '/ckpt_' + task_names)
        vars_list.append(myModel.get_weights())
        # 重置model参数为初始值
        myModel.load_weights(ckpt_path_theta_0)

    # 更新模型初始化参数theta_0
    update_vars(myModel, vars_list, e)
    myModel.save_weights(ckpt_path_theta_0)

    # vali_data_path = 'data_tasks/' + vali_sample
    # vali_train_batches = get_batches_v2(vali_data_path, batch_size=200, batch_num=3, taskname=vali_sample)

    # ===========验证阶段训练NER模型=============
    vali_train_batches = get_batches_v4(train_data_path_list=vali_train_data_paths, batch_size=batch_size, batch_num=1,
                                        taskname=validation_tasks)
    vali_train_batches_pred = bert_embedding(vali_train_batches)
    # 指数衰减学习率
    exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.005, decay_steps=1, decay_rate=0.96)
    myModel.optimizer = tf.optimizers.Adam(exponential_decay)

    with tqdm(total=inner_iters) as bar:
        maxF1 = 0
        for i in range(inner_iters):
            _,F1_t = myModel.inner_train_one_step(vali_train_batches_pred, inner_iters=inner_iters, inner_epochNum=i,
                                         outer_epochNum=epoch,
                                         task_name=validation_tasks, log_writer=log_writer_vali_train)
            if F1_t>maxF1:
                maxF1 = F1_t
                myModel.save_weights(ckpt_dir_theta_t)
            bar.update(1)
    # 在新数据上测试效果
    print('**********************************************\n validation in task:{}\n'
          '**********************************************\n'.format(validation_tasks))

    # 验证阶段 测试NER模型
    test_loss, pred_tags_masked, tag_ids_padded,P, R, F1 = myModel.validate_one_batch(vali_test_batch_pred, validation_tasks,
                                                                             log_writer_vali_test, epoch)

    # 记录epoch
    Record_epoch_num(recordFileName, epoch)

    endtime = time.time()
    print('done inner epoch:{}!——————run time ：{}s.\n'.format(i, str(endtime - starttime)))
