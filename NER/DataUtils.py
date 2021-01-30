import io
import json
import os, re
import pickle
import random

import Data_processing

tag2id = {'<pad>': 0,
          'O': 1,
          'S-Person': 2, 'B-Person': 3, 'I-Person': 4, 'E-Person': 5,
          'S-Org': 6, 'B-Org': 7, 'I-Org': 8, 'E-Org':9,
          'S-Loc': 10, 'B-Loc': 11, 'I-Loc': 12, 'E-Loc': 13,
          'S-Time': 14, 'B-Time': 15, 'I-Time': 16, 'E-Time': 17,
          }

RMRB_tag = {
    "nr": "Person",
    "ns": "Loc",
    "nt": "Org",
    "t": "Time"
}




# 人民日报语料库专用函数
def textReplace(text):
    """
    去除文中[ ]复合标签内的多余标签，只取外层标签
    :param text: 原始文本字符串
    :return:  替换修改后的文本字符串
    """
    # 去除[]中多余的标签
    flist = re.findall(r'/[a-z]{3,5}', text)
    flist = set(flist)
    print(flist)

    def change(str):
        str = str.group()
        res = re.sub(r'/[a-z]{3,6}', '', str)
        return res

    newtext = re.sub('\[.*?\]', change, text)
    return newtext

# 人民日报语料库专用 转换BIOES：
def text2BIOES(path, outpath):
    '''
    1词1行的文本转换成BIOES标注的
    :param path:
    :return:
    '''
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    BIOES_content = ""
    for i,line in enumerate(lines):
        line = line.strip()
        if line is not '':
            try:
                word, tag = line.split('/')
                tag = tag_change(tag)
                if tag is 'O':
                    for char in word:
                        BIOES_content += char + '\tO' + '\n'
                else:
                    if len(word) == 1:
                        BIOES_content += word + '\tS-' + tag + '\n'
                    elif len(word) > 1:
                        BIOES_content += word[0] + '\tB-' + tag + '\n'
                        for char in word[1:-1]:
                            BIOES_content += char + '\tI-' + tag + '\n'
                        BIOES_content += word[-1] + '\tE-' + tag + '\n'
            except Exception as e:
                print(e)
                print('=========this line is:',i,':', line)
                word, _, tag = line.split('/')
                BIOES_content += '/\tO' + '\n'
        else:
            BIOES_content += '\n'
    with open(outpath, 'w', encoding='utf-8') as f:
        f.write(BIOES_content)
    print('done!->', path)

# 人民日报语料库专用 原标注转换为需要的标注
def tag_change(tag):
    """
    原标注转换为需要的标注
    :param tag:
    :return:
    """
    if tag in RMRB_tag.keys():
        tag = RMRB_tag[tag]
    else:
        tag = 'O'
    return tag

# 数据中句子最大长度
def maxlen(data):
    '''

    :param data: data=[[sentence],[sentence],....]
             sentence=[chars]
    :return:
    '''
    return max([len(s) for s in data ])

# 读取BIOES数据，转换为模型所需的列表
def read_train_data(traindata_path,vocab_path):
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
                if char.isdigit():
                    char ='<NUM>'
                elif ('\u0041' <= char <= '\u005a') or ('\u0061' <= char <= '\u007a'):
                    char = '<ENG>'
                chars.append(char)
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

# 读取BIOES数据，仅测试用
# def read_data_for_test(traindata_path):
#     '''
#     BIOES标注好的文本 读取后转换为模型所需列表
#     :param traindata_path: BIOES标注好的文本路径
#     :return: data=[[sentence],[sentence],....]
#             sentence=[[chars],[charids],[tags],[tag_ids]]
#     '''
#     data = []
#     with io.open(traindata_path, encoding='utf-8') as fr:
#         lines = fr.readlines()
#
#     chars, tags = [], []
#     for i,line in enumerate(lines):
#         print(i,'--->',traindata_path)
#         if line != '\n':
#             # [char, label] = line.strip().split()
#             try:
#                 char = ''.join(line).strip().split()[0]
#                 chars.append(char)
#                 tag = ''.join(line).strip().split()[1]
#                 tags.append(tag)
#             except Exception:
#                 print(line)
#         else:
#             if len(chars) < 1 or len(tags) < 1:
#                 continue
#             data.append(chars)
#             chars,  tags = [], []
#     print('::::::::::::::::::::::::::::::::', len(data))
#     # print(data)
#     return data

# 建立训练数据batches
def get_batches(data, batch_size):
    '''
    :param data: data=[[sentence],[sentence],....]
             sentence=[[chars],[charids],[tags],[tag_ids]]
    :param batch_size:
    :return:
    '''
    num_batches = len(data) // batch_size
    print(num_batches)
    random.shuffle(data)
    batches = []
    for i in range(num_batches):
        batches.append(data[i*batch_size:(i+1)*batch_size])
    return batches



# for NER
if __name__ =='__main__':
    tpath = r'F:\zzd\毕业论文\论文代码\NER\data\someNEWS_BIOES.dev'
    vpath = r'F:\zzd\毕业论文\论文代码\NER\vocab\vocab.pkl'
    # data = read_train_data(tpath,vpath)
    # batches = get_batches(data,100)
    # print(len(batches))
    with io.open(tpath,encoding='utf-8') as f:
        lines = f.readlines()
    print(lines[0])