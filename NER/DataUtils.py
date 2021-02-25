import io
import json
import os, re
import pickle
import random
import numpy as np

tag2id = {'<pad>': 0,
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
def read_train_data(traindata_path, vocab_path):
    '''
    BIOES标注好的文本 读取后转换为模型所需列表
    :param traindata_path: BIOES标注好的文本路径
    :return: data=[[sentence],[sentence],....]
            sentence=[[chars],[charids],[tags],[tag_ids]]
    '''
    data = []
    with io.open(traindata_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    with open(vocab_path, 'rb') as f:
        char2id = pickle.load(f)

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
                tag = ''.join(line).strip().split()[1]
                tags.append(tag)
                tag_id = tag2id[tag]
                tag_ids.append(tag_id)
            except Exception as e:
                print(line)
                print(e)
        else:
            if len(chars) < 1 or len(tags) < 1:
                continue
            data.append((chars, charids, tags, tag_ids))
            chars, charids, tags, tag_ids = [], [], [], []
    print(traindata_path, ':', len(data))
    # print(data)
    return data


# 建立训练数据batches
def get_batches(data, batch_size):
    '''
    :param data: data=[[sentence],[sentence],....]
             sentence=[[chars],[charids],[tags],[tag_ids]]
    :param batch_size:
    :return: [[one batch],[]...]
            one batch: [[charids_list],[tagids_list]]
    '''
    num_batches = len(data) // batch_size
    print(num_batches)
    random.shuffle(data)
    batches = []
    for i in range(num_batches):
        batches.append(data[i*batch_size:(i+1)*batch_size])
    return batches


# 提取真实和预测的 有效序列标签 标签的偏移和混淆都不计入提取范围
def findall_tag(tag_ids,seq_len_list):
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
            if i+1 < len(tags_flatten) and (tag <= tags_flatten[i + 1] <= tag + 1):
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

# 解析标签id
def get_id2tag(tag_list):
    '''

    :param tag_list:  [batchsize, seq_length]
    :return:
    '''

    list = np.array(tag_list).flatten()
    id2tag = {tag2id[k]:k for k in tag2id.keys()}
    newtag_list = []
    for i in range(len(list)):
        if list[i]!=0:
            newtag_list.append(id2tag[list[i]])
    return newtag_list,list


# 计算P、R、F1值   标签的偏移和混淆都不计入正确预测的范围
def P_R_F1_score(p_dict: dict, t_dict: dict) -> (float,float,float):
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

    if P+R==0:
        F1 = 0.0
    else:
        F1 = 2 * P * R / (P + R)

    return (P,R,F1)




#  test
if __name__ == '__main__':
    tpath = r'F:\zzd\毕业论文\论文代码\NER\data\someNEWS_BIOES.dev'
    vpath = r'F:\zzd\毕业论文\论文代码\NER\vocab\vocab.pkl'
    # data = read_train_data(tpath,vpath)
    # batches = get_batches(data,100)
    # print(len(batches))
    with io.open(tpath, encoding='utf-8') as f:
        lines = f.readlines()
    print(lines[0])
