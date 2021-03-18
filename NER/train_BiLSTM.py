import datetime
from Mymodel_V3 import Model_NER,conf
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

train_tasks = ['address', 'scene', 'government', 'organization',
               'company', 'name', 'book']
# 训练集采集
task_samples = random.sample(train_tasks, 3)
task_samples.sort()
task_names = '-'.join(task_samples)
train_data_path_list = []
for t in task_samples:
    p = os.path.join('data_tasks', t)
    train_data_path_list.append(p)

train_batch_num = 1
vocab_path = 'vocab/vocab.pkl'
train_batches = get_batches_v3(train_data_path_list,200, batch_num=train_batch_num, taskname=task_samples)
# 获取验证集数据： 验证集分两部分，验证_训练集 和 验证_测试集
vali_train_data_paths = []
vali_test_data_path = ['data/CLUE_BIOES_dev']
vali_test_batches = get_batches_v3(train_data_path_list=vali_test_data_path, batch_size=200, batch_num=1,
                                    taskname=task_samples)
print('batches is ready!\n'
      '-----------------------------------------------------------\n')

task = 'CLUE_ALL'
recordFileName = 'record_bilstm_' + str(train_batch_num*3) + 'b-' + task_names
create_record_dirs(recordFileName)
epochNum = get_epochNum(recordFileName)  # 获取当前记录的epoch数

# 配置模型参数、检查点
configers = conf(choose_mod='BiLSTM')
myModel_instance = Model_NER(configers)
ckpt_dir = os.path.join(recordFileName, 'checkpoints')
checkpoint = tf.train.Checkpoint(optimizer=myModel_instance.optimizer, model=myModel_instance)
ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=ckpt_dir, max_to_keep=5)

if epochNum != 0:
    # 加载checkpoints
    latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
    ckp = tf.train.Checkpoint(model=myModel_instance)
    ckp.restore(latest_ckpt)

# 配置tensorboard
# log_dir_train = recordFileName + '/tensorboard/' + datetime.datetime.now().strftime("%Y%m%d-%H%M") + '-train'
# log_dir_train = recordFileName + '/tensorboard/' + '-train'
log_dir_train = recordFileName + '/tensorboard/' + '-train'
log_dir_test = recordFileName + '/tensorboard/' + '-test'
log_writer_train = tf.summary.create_file_writer(log_dir_train)
log_writer_test = tf.summary.create_file_writer(log_dir_test)



for epoch in range(epochNum, epochNum + 300):
    starttime = time.time()
    myModel_instance.inner_train_one_step(train_batches, inner_iters=0,inner_epochNum=epoch,outer_epochNum=0, task_name=task_samples, log_writer=log_writer_train)
    endtime = time.time()
    print('\ndone epoch:{}!  task:{}——————run time ：'.format(epoch, task_samples), str(endtime - starttime), 's.')
    print('===============================================================================\n')
    ckpt_manager.save(checkpoint_number=epoch)

    # test_batch = random.choice(test_batches)
    test_loss, pred_tags_masked, tag_ids_padded = myModel_instance.predict_one_batch(vali_test_batches[0])
    p_tags_char, _ = get_id2tag(pred_tags_masked,taskname=task_samples)
    t_tags_char, _ = get_id2tag(tag_ids_padded,taskname=task_samples)
    (P, R, F1),label_result = evaluate(t_tags_char, p_tags_char, verbose=True)
    write_to_log(test_loss,P,R,F1,label_result,log_writer_test,epoch)


    # 记录epoch
    Record_epoch_num(recordFileName, epoch)

