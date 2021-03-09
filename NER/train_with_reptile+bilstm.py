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

recordFileName = 'test_record01'
create_record_dirs(recordFileName)
epochNum = get_epochNum(recordFileName)  # 获取当前记录的epoch数

# 配置模型参数、检查点
configers = conf()
myModel = Model_NER(configers)
ckpt_dir_inner = os.path.join(recordFileName, 'checkpoints')
ckpt_dir_theta_0 = os.path.join(recordFileName, 'theta_0')
ckpt_path_theta_0 = os.path.join(ckpt_dir_theta_0, 'ckpt_theta_0')
ckpt_dir_theta_t = os.path.join(recordFileName, 'theta_t')
ckpt_path_theta_t = os.path.join(ckpt_dir_theta_t, 'ckpt_theta_t')
checkpoint = tf.train.Checkpoint(optimizer=myModel.optimizer, model=myModel)
ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=ckpt_dir_inner, max_to_keep=5)

if epochNum == 0:
    myModel.save_weights(ckpt_path_theta_0)
else:
    myModel.load_weights(ckpt_path_theta_t)

# 配置tensorboard
# log_dir_train = recordFileName + '/tensorboard/' + datetime.datetime.now().strftime("%Y%m%d-%H%M") + '-train'
# log_dir_test = recordFileName + '/tensorboard/' + datetime.datetime.now().strftime("%Y%m%d-%H%M") + '-test'
log_dir_train = recordFileName + '/tensorboard/' + '-train'
log_dir_vali_train = recordFileName + '/tensorboard/' + '-vali_train'
log_dir_vali_test = recordFileName + '/tensorboard/' + '-vali_test'
log_writer_train = tf.summary.create_file_writer(log_dir_train)
log_writer_vali_train = tf.summary.create_file_writer(log_dir_vali_train)
log_writer_vali_test = tf.summary.create_file_writer(log_dir_vali_test)

print('batches is ready!\n'
      '-----------------------------------------------------------\n')

train_tasks = ['address', 'scene', 'government', 'organization',
               'position', 'name', 'book', 'movie']
validation_tasks = ['game', 'company']

inner_iters = 20
e = 0.1

for epoch in range(epochNum, epochNum + 1000):
    starttime = time.time()
    print('====outer epoch:{}==========================================='.format(epoch))
    vars_list = []
    task_samples = random.sample(train_tasks, 5)
    for task in task_samples:
        print('task:{}================'.format(task))
        if epoch == 0:
            myModel.load_weights(ckpt_path_theta_0)
        else:
            myModel.load_weights(ckpt_path_theta_t)
        train_data_path = 'data_tasks/' + task
        batches = get_batches_v2(train_data_path, batch_size=200, batch_num=3, taskname=task)
        with tqdm(total=inner_iters) as bar:
            for i in range(inner_iters):
                myModel.inner_train_one_step(batches, inner_iters, i, epochNum,
                                             task_name=task, log_writer=log_writer_train)
                bar.update(1)

        myModel.save_weights(ckpt_dir_inner + '/ckpt_' + task)
        vars_list.append(myModel.get_weights())

    # 更新模型初始化参数
    update_vars(myModel, vars_list, e)
    myModel.save_weights(ckpt_path_theta_t)

    vali_sample = random.choice(validation_tasks)
    vali_data_path = 'data_tasks/' + vali_sample
    vali_train_batches = get_batches_v2(vali_data_path, batch_size=200, batch_num=3, taskname=vali_sample)

    with tqdm(total=inner_iters) as bar:
        for i in range(inner_iters):
            myModel.inner_train_one_step(batches, inner_iters=inner_iters, inner_epochNum=i, outer_epochNum=epochNum,
                                         task_name=vali_sample, log_writer=log_writer_vali_train)
            bar.update(1)
    # 在新数据上测试效果
    print('**********************************************\n validation in task:{}\n'
          '**********************************************\n'.format(vali_sample))

    vali_test_batches = get_batches_v2(vali_data_path, batch_size=200, batch_num=3, taskname=vali_sample)
    vali_test_batche = vali_test_batches[1]
    test_loss, pred_tags_masked, tag_ids_padded = myModel.predict_one_batch(vali_test_batche)
    p_tags_char, _ = get_id2tag(pred_tags_masked, taskname=vali_sample)
    t_tags_char, _ = get_id2tag(tag_ids_padded, taskname=vali_sample)
    (P, R, F1), label_result = evaluate(t_tags_char, p_tags_char, verbose=True)
    write_to_log(test_loss, P, R, F1, label_result, log_writer_vali_test, epochNum)

    # 记录epoch
    Record_epoch_num(recordFileName, epoch)

    endtime = time.time()
    print('done inner epoch:{}!——————run time ：{}s.\n'.format(i, str(endtime - starttime)))
