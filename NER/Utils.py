import io
import json
import os, re
import pickle
import random
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

tag2id_v0 = {'<pad>': 0,
             'O': 1,
             'S-Person': 2, 'B-Person': 3, 'I-Person': 4, 'E-Person': 5,
             'S-Org': 6, 'B-Org': 7, 'I-Org': 8, 'E-Org': 9,
             'S-Loc': 10, 'B-Loc': 11, 'I-Loc': 12, 'E-Loc': 13,
             'S-Time': 14, 'B-Time': 15, 'I-Time': 16, 'E-Time': 17,
             }


RMRB_tag = {
    "nr": "Person",
    "ns": "Loc",
    "nt": "Org",
    "t": "Time"
}


# 读取词表 返回字典（字->id）
def read_vocab(vocab_path):
    '''
    :param vocab_path:
    :return:                返回一个字典
    '''
    # vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


# 数据中句子列表中句子的最大长度
def maxlen(sentencs):
    '''

    :param data: sentencs=[[sentence],[sentence],....]
             sentence=[[char1],[char2],...[charn]]
    :return:
    '''
    return max([len(s) for s in sentencs])


# 读取BIOES数据，转换为模型所需的列表
def read_train_data(traindata_path, vocab_path, taskname=''):
    '''
    BIOES标注好的文本 读取后转换为模型所需列表
    :param traindata_path: BIOES标注好的文本路径
    :param vocab_path: data=[[sentence],[sentence],....]
                     sentence=[[chars],[charids],[tags],[tag_ids]]
    :param taskname: 根据任务类别 只取数据中对应类别的标签，其他标签置为O
    :return:
    '''
    data = []
    with io.open(traindata_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    with open(vocab_path, 'rb') as f:
        char2id = pickle.load(f)

    tag2id = {
        '<pad>': 0,
        'O': 1,
        'S-' + taskname: 2, 'B-' + taskname: 3, 'I-' + taskname: 4, 'E-' + taskname: 5,
    }
    chars, charids, tags, tag_ids = [], [], [], []
    for line in lines:
        if line != '\n':
            # [char, label] = line.strip().split()
            try:
                char = ''.join(line).strip().split()[0]
                chars.append(char)
                if char.isdigit():
                    char = '<NUM>'
                elif ('\u0041' <= char <= '\u005a') or ('\u0061' <= char <= '\u007a'):
                    char = '<ENG>'

                charid = char2id[char]
                charids.append(charid)
                if taskname == '':
                    tag = ''.join(line).strip().split()[1]
                    tags.append(tag)
                    tag_id = tag2id_v0[tag]
                    tag_ids.append(tag_id)
                else:
                    tag = ''.join(line).strip().split()[1]
                    tags.append(tag)
                    tag_id = tag2id[tag] if tag in tag2id.keys() else 1
                    tag_ids.append(tag_id)

            except Exception as e:
                print(line)
                print('find error!!!!')
                print(e)
        else:
            if len(chars) < 1 or len(tags) < 1:
                continue
            data.append((chars, charids, tags, tag_ids))
            chars, charids, tags, tag_ids = [], [], [], []
    print('------>已读取数据源 {} ! 其中包含{}个句子'.format(traindata_path, len(data)))
    # print(data)
    return data

# 原始data转换成batches
def data_to_batches(data, batch_size, batch_num):
    '''
    :param data: data=[[sentence],[sentence],....]
             sentence=[[chars],[charids],[tags],[tag_ids]]
    :param batch_size:
    :return: [[one batch],[]...]
            one batch: [[charids_list],[tagids_list]]
    '''
    num_batches = len(data) // batch_size
    random.shuffle(data)
    batches = []
    for i in range(batch_num):
        batches.append(data[i * batch_size:(i + 1) * batch_size])
    return batches


# 获得训练batches V1
def get_batches_v1(train_data_dir, data_file_Num, batchsize):
    '''
    从data_split目录中选取 已划分为200句一个文件的训练数据
    :param train_data_dir:
    :param data_file_Num: 从源训练数据文件中读取的文件数量（读取的所有训练文件 制作为一个batches）
    :param batchsize:
    :return:
    '''
    train_files = os.listdir(train_data_dir)

    # 准备数据
    datas = []
    vocab_path = 'vocab/vocab.pkl'
    for i in range(data_file_Num):
        train_data_path = os.path.join(train_data_dir, train_files[i])
        data = read_train_data(train_data_path, vocab_path)
        datas.extend(data)
    batch_num = len(datas) // batchsize
    batches = data_to_batches(datas, batch_size=batchsize, batch_num=batch_num)
    return batches


# 获得训练batches V2
def get_batches_v2(train_data_path, batch_size, batch_num, taskname=None):
    vocab_path = 'vocab/vocab.pkl'
    data = read_train_data(train_data_path, vocab_path, taskname)
    batches = data_to_batches(data, batch_size, batch_num)
    return batches


# 从批数据中 分离出seq_ids 和 tag_ids,以及每一句的长度列表：seq_len_list
def get_train_data_from_batch(batch):
    '''
    :param batch: [[sentence],[sentence],....]
            sentence=[[chars],[charids],[tags],[tag_ids]]
    :return:
    '''
    seq_ids, tag_ids = [], []
    for sentence in batch:
        seq_ids.append(sentence[1])
        tag_ids.append(sentence[3])
    seq_len_list = [len(s) for s in seq_ids]
    seq_ids = keras.preprocessing.sequence.pad_sequences(seq_ids, maxlen=max(seq_len_list), padding='post', value=0)
    seq_ids = keras.preprocessing.sequence.pad_sequences(seq_ids, maxlen=max(seq_len_list) + 1, padding='post', value=0)
    tag_ids = keras.preprocessing.sequence.pad_sequences(tag_ids, maxlen=max(seq_len_list), padding='post', value=0)
    tag_ids = keras.preprocessing.sequence.pad_sequences(tag_ids, maxlen=max(seq_len_list) + 1, padding='post', value=0)
    seq_ids = tf.convert_to_tensor(seq_ids, dtype='int32')
    tag_ids = tf.convert_to_tensor(tag_ids, dtype='int32')
    return (seq_ids, tag_ids, seq_len_list)


# 将序列实际长度之后的位变为0
def seq_masking(seqs, lens):
    '''
    将序列实际长度之后的位变为0
    :param seqs: [seqs_num, max_seq_len]
    :param lens: [seqs_num] 每一句的实际长度
    :return: [seqs_num, max_seq_len] 实际长度以后的位置都为0
    '''
    maxl = max(lens) + 1
    mask = tf.sequence_mask(lengths=lens, maxlen=maxl, dtype=tf.int32)
    seqs_masked = seqs * mask
    return seqs_masked


# 提取真实和预测的 有效序列标签 标签的偏移和混淆都不计入提取范围
def findall_tag(tag_ids, seq_len_list):
    '''
    只识别连续的BIE、和S标注的位置
    :param tag_ids: [batchesize, seq_length]
    :return: 返回一个字典 key：一串序列索引组成的字符串，来表示被标注实体的位置 如 ‘55.56.57’ 表示此实体在序列中的索引为55-57。
                      value：标注最后一个字的标注号，如，被标注‘E-Person’或‘S-Org 则此项值为5或9
    '''
    tag_dict = {}
    tags_flatten = np.array(tag_ids).flatten()
    key = ''
    for i, tag in enumerate(tags_flatten):

        if tags_flatten[i] == 0:
            key = ''
            continue
        if tag % 4 == 2:
            key = str(i)
            tag_dict[key] = tag
            key = ''
        elif tag % 4 == 3 and (tag + 1 <= tags_flatten[i + 1] <= tag + 2):
            key = str(i) + '.'
        elif tag % 4 == 0 and tag != 0:
            if i + 1 < len(tags_flatten) and (tag <= tags_flatten[i + 1] <= tag + 1):
                key += str(i) + '.'
            else:
                key = ''
        elif tag % 4 == 1 and key is not '':
            key += str(i) + '.'
            tag_dict[key] = tag
            key = ''
        else:
            key = ''
    return tag_dict


# 解析标签id-->输出一个flatten列表
def get_id2tag(tag_list, taskname=''):
    '''

    :param tag_list:  [batchsize, seq_length]
    :return: 一维list
    '''

    tag2id = {
        '<pad>': 0,
        'O': 1,
        'S-' + taskname: 2, 'B-' + taskname: 3, 'I-' + taskname: 4, 'E-' + taskname: 5,
    }

    tag_list_flatten = np.array(tag_list).flatten()
    if taskname=='':
        id2tag = {tag2id_v0[k]: k for k in tag2id_v0.keys()}
    else:
        id2tag = {tag2id[k]: k for k in tag2id.keys()}
    newtag_list = []
    for i in range(len(tag_list_flatten)):
        if tag_list_flatten[i] != 0:
            if tag_list_flatten[i] in id2tag.keys():
                newtag_list.append(id2tag[tag_list_flatten[i]])
            else:
                newtag_list.append('O')
    return newtag_list, tag_list_flatten


# 计算P、R、F1值   标签的偏移和混淆都不计入正确预测的范围
def P_R_F1_score(p_dict: dict, t_dict: dict) -> (float, float, float):
    '''
    :param p_dict: 所有预测出的标签的字典
    :param t_dict: 所有真实标签的字典
    :return: (P,R,F1)
    '''
    count = 0.0
    P = -1.0
    for k in p_dict.keys():
        if k in t_dict.keys() and p_dict[k] == t_dict[k]:
            count += 1.0

    if len(p_dict) == 0:
        P = 0.0
    else:
        P = count / len(p_dict.keys())

    if len(t_dict) == 0:
        R = 0.0
    else:
        R = count / len(t_dict.keys())

    if P + R == 0:
        F1 = 0.0
    else:
        F1 = 2 * P * R / (P + R)

    return (P, R, F1)


# 查找已记录的epoch数
def get_epochNum(recordFileName):
    '''
    若不存在则创建记录目录，并初始化epochNum为0
    :param recordFileName:
    :return:
    '''
    epoch_dir = os.path.join(recordFileName, 'epoch_record')
    if os.path.exists(epoch_dir):
        epoch_num = int(os.listdir(epoch_dir)[0])
    else:
        os.mkdir(epoch_dir)
        epochFilePath = os.path.join(epoch_dir, '0')
        fd = open(epochFilePath, 'w')
        fd.close()
        epoch_num = 0
    return epoch_num


# 记录全局epoch
def Record_epoch_num(recordFileName, epoch_num):
    '''
    记录全局epoch
    :param epoch_dir: 记录epoch文件的目录
    :param epoch_num: 之前epoch值
    :return:  当前epoch值
    '''
    epoch_dir = os.path.join(recordFileName, 'epoch_record')
    os.chdir(epoch_dir)
    new_num = epoch_num + 1
    os.rename(str(epoch_num), str(new_num))
    epoch_num = new_num
    os.chdir('..')
    os.chdir('..')
    return epoch_num


# 创建实验记录的文件目录
def create_record_dirs(recordname):
    if not os.path.exists(recordname):
        checkpoints_dir = recordname + '/checkpoints'
        init_theta_dir = recordname + '/theta_0'
        target_theta_dir = recordname + '/theta_t'
        tensorboard_dir = recordname + '/tensorboard'
        li = [checkpoints_dir, init_theta_dir, target_theta_dir, tensorboard_dir]
        for p in li:
            os.makedirs(p)


# 记录测试的P R F1值
def write_to_log(loss, P, R, F1, label_dict, log_writer, epochNum):
    with log_writer.as_default():
        tf.summary.scalar("loss", loss, step=epochNum)
        tf.summary.scalar("P", P, step=epochNum)
        tf.summary.scalar("R", R, step=epochNum)
        tf.summary.scalar("F1", F1, step=epochNum)


# ====================functions in Reptile=========================

# 计算各任务训练参数的平均值
def average_vars(vars_list):
    res = []
    for variables in zip(*vars_list):
        res.append(np.mean(variables, axis=0))
    return res


# outer loop 更新参数
def update_vars(myModel, vars_list, epsilon):
    t_var = average_vars(vars_list)
    oldvar = myModel.get_weights()
    newvar = [v2 + epsilon * (v1 - v2) for v1, v2 in zip(t_var, oldvar)]
    myModel.set_weights(newvar)

if __name__ == '__main__':
    path = r'F:\zzd\毕业论文\论文代码\NER\data_tasks\address'
    vocabpath = r'F:\zzd\毕业论文\论文代码\NER\vocab\vocab.pkl'
    batches = get_batches_v2(path,batch_size=200,batch_num=3,taskname='address')
    batches = 1