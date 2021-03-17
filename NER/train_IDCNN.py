import datetime
from Mymodel_base import Model_NER,conf
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

tasks = ['book','movie','game','company']
train_batch_num = 20
for task in tasks:

    recordFileName = 'record_idcnn_' + str(train_batch_num) + 'b-' + task
    create_record_dirs(recordFileName)
    epochNum = get_epochNum(recordFileName)  # 获取当前记录的epoch数

    # 配置模型参数、检查点
    configers = conf(choose_mod='IDCNN')
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
    log_dir_train = recordFileName + '/tensorboard/' + '-train'
    log_dir_test = recordFileName + '/tensorboard/' + '-test'
    log_writer_train = tf.summary.create_file_writer(log_dir_train)
    log_writer_test = tf.summary.create_file_writer(log_dir_test)

    # 获取数据
    train_data_path = 'data_tasks/' + task
    test_data_path = 'data/CLUE_BIOES_dev'
    vocab_path = 'vocab/vocab.pkl'
    train_batches = get_batches_v1(train_data_path,200, taskname=task)
    train_batches=train_batches[0:train_batch_num]
    test_batches = get_batches_v1(test_data_path,200, taskname=task)

    print('batches is ready!\n'
          '-----------------------------------------------------------\n')


    P_max = 0.0
    for epoch in range(epochNum, epochNum + 200):
        starttime = time.time()
        myModel_instance.inner_train_one_step(train_batches, epochNum=epoch, task_name=task, log_writer=log_writer_train, log_dir=log_dir_train)
        endtime = time.time()
        print('===============================================================================\n')
        print('\ndone epoch:{}!——————run time ：'.format(epoch), str(endtime - starttime), 's.')
        print('===============================================================================')

        test_batch = test_batches[1]
        test_loss, pred_tags_masked, tag_ids_padded = myModel_instance.predict_one_batch(test_batch)
        p_tags_char, _ = get_id2tag(pred_tags_masked,taskname=task)
        t_tags_char, _ = get_id2tag(tag_ids_padded,taskname=task)
        (P, R, F1),label_result = evaluate(t_tags_char, p_tags_char, verbose=True)
        write_to_log(test_loss,P,R,F1,label_result,log_writer_test,epoch)

        if P_max<P:
            ckpt_manager.save(checkpoint_number=epoch)
            P_max = P

        # 记录epoch
        Record_epoch_num(recordFileName, epoch)

#     starttime = time.time()
# tf.test.is_gpu_available()
# train_data_path = r'F:\zzd\毕业论文\论文代码\NER\data\someNEWS_BIOES.dev'
# vocab_path = r'F:\zzd\毕业论文\论文代码\NER\vocab\vocab.pkl'
# data = read_train_data(train_data_path, vocab_path)
# batches = get_batches(data, 100)
# batch = batches[0]
# seq_ids, tag_ids, seq_len_list = get_train_data_from_batch(batch)
# tagdict = findall_tag(tag_ids)
# print(tagdict)
