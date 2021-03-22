import datetime
from Mymodel_V3 import Model_NER, conf
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

mod = 'BiLSTM'
inner_iters = 8
e = 0.005
e_final = 0.001
batch_size = 200
recordFileName = '_'.join(['test_3L_MAML+' + mod, str(inner_iters) + 'i', str(e)+'-' + str(e_final) + 'e',str(batch_size)+'bs'])
create_record_dirs(recordFileName)
epochNum = get_epochNum(recordFileName)  # 获取当前记录的epoch数

# 配置模型参数、检查点
configers = conf(choose_mod=mod)
myModel = Model_NER(configers)
ckpt_dir_inner = os.path.join(recordFileName, 'checkpoints')
ckpt_dir_theta_0 = os.path.join(recordFileName, 'theta_0')
ckpt_path_theta_0 = os.path.join(ckpt_dir_theta_0, 'ckpt_theta_0')
# ckpt_dir_theta_t = os.path.join(recordFileName, 'theta_t')
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

vali_test_data_path = 'data/CLUE_BIOES_dev'
vali_test_batches = get_batches_v1(train_data_path=vali_test_data_path, batchsize=200,
                                    taskname=validation_tasks)
vali_test_batch = []
for a in vali_test_batches:
    vali_test_batch.extend(a)

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
        myModel.optimizer = tf.optimizers.Adam(learning_rate=0.005)

        # ==============内循环训练阶段===================
        batches = get_batches_v3(train_data_path_list=path_list, batch_size=batch_size, batch_num=1, taskname=task_samples)
        with tqdm(total=inner_iters) as bar:
            for i in range(inner_iters):
                train_loss, train_P = myModel.inner_train_one_step(batches, inner_iters, i, epoch,
                                                                   task_name=task_samples, log_writer=log_writer_train)
                bar.update(1)
            print('train_loss:{}   train_Precision:{}'.format(train_loss, train_P))

        # 记录当前任务训练所得model的参数
        task_names = '-'.join(task_samples)
        # myModel.save_weights(ckpt_dir_inner + '/ckpt_' + task_names)
        vars_list.append(myModel.get_weights())
        # 重置model参数为初始值
        myModel.load_weights(ckpt_path_theta_0)

    # TODO: 按MAML方式 求二次梯度 更新模型初始化参数theta_0
    update_vars(myModel, vars_list, e)
    myModel.save_weights(ckpt_path_theta_0)

    # vali_data_path = 'data_tasks/' + vali_sample
    # vali_train_batches = get_batches_v2(vali_data_path, batch_size=200, batch_num=3, taskname=vali_sample)

    # ===========验证阶段训练NER模型=============
    vali_train_batches = get_batches_v3(train_data_path_list=vali_train_data_paths, batch_size=batch_size, batch_num=1,
                                        taskname=validation_tasks)
    myModel.optimizer = tf.optimizers.Adam(learning_rate=0.005)
    with tqdm(total=inner_iters) as bar:
        for i in range(inner_iters):
            myModel.inner_train_one_step(vali_train_batches, inner_iters=inner_iters, inner_epochNum=i, outer_epochNum=epoch,
                                         task_name=validation_tasks, log_writer=log_writer_vali_train)
            bar.update(1)
    # 在新数据上测试效果
    print('**********************************************\n validation in task:{}\n'
          '**********************************************\n'.format(validation_tasks))

    # 验证阶段 测试NER模型
    test_loss, pred_tags_masked, tag_ids_padded = myModel.predict_one_batch(vali_test_batch)
    p_tags_char, _ = get_id2tag(pred_tags_masked, taskname=validation_tasks)
    t_tags_char, _ = get_id2tag(tag_ids_padded, taskname=validation_tasks)
    (P, R, F1), label_result = evaluate(t_tags_char, p_tags_char, verbose=True)
    write_to_log(test_loss, P, R, F1, label_result, log_writer_vali_test, epoch)

    # 记录epoch
    Record_epoch_num(recordFileName, epoch)

    endtime = time.time()
    print('done inner epoch:{}!——————run time ：{}s.\n'.format(i, str(endtime - starttime)))
