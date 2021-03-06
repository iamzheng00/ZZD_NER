import datetime
from Mymodel_V2 import Model_bilstm,conf
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
myModel = Model_bilstm(configers)
ckpt_dir_inner = os.path.join(recordFileName, 'checkpoints')
ckpt_dir_theta_0 = os.path.join(recordFileName, 'theta_0')
ckpt_dir_theta_t = os.path.join(recordFileName, 'theta_t')
checkpoint = tf.train.Checkpoint(optimizer=myModel.optimizer, model=myModel)
ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=ckpt_dir_inner, max_to_keep=5)


# 加载checkpoints
# latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
# ckp = tf.train.Checkpoint(model=model)
# ckp.restore(latest_ckpt)

# 配置tensorboard
# log_dir_train = recordFileName + '/tensorboard/' + datetime.datetime.now().strftime("%Y%m%d-%H%M") + '-train'
log_dir_train = recordFileName + '/tensorboard/' + '-train'
# log_dir_test = recordFileName + '/tensorboard/' + datetime.datetime.now().strftime("%Y%m%d-%H%M") + '-test'
log_dir_test = recordFileName + '/tensorboard/' + '-test'
log_writer_train = tf.summary.create_file_writer(log_dir_train)
log_writer_test = tf.summary.create_file_writer(log_dir_test)

# 获取数据
train_data_dir = 'data_split/MSRA'
batches = get_batches_v1(train_data_dir, 10, configers.batchsize)
train_batches = batches[0:3]
test_batch = batches[5]
print('batches is ready!\n'
      '-----------------------------------------------------------\n')


for epoch in range(epochNum + 200):
    starttime = time.time()
    myModel.inner_train_one_step(train_batches, inner_epochNum=epoch, ckpt_manager=ckpt_manager, log_writer=log_writer_train)
    endtime = time.time()
    print('done one inner epoch!——————run time ：', str(endtime - starttime), 's.')

    test_loss, pred_tags_masked, tag_ids_padded = myModel.predict_one_batch(test_batch)
    p_tags_char, _ = get_id2tag(pred_tags_masked)
    t_tags_char, _ = get_id2tag(tag_ids_padded)
    (P, R, F1),label_result = evaluate(t_tags_char, p_tags_char, verbose=True)
    write_to_log(test_loss,P,R,F1,label_result,log_writer_test,epochNum)


    # 记录epoch
    epochNum = Record_epoch_num(recordFileName, epochNum)

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
