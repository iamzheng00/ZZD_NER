import io
import os
import pickle
import re

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
tag_trans={
    'PER':'Person',
    'ORG':'Org',
    'LOC':'Loc'
}

# 人民日报语料库专用函数 处理原始语料库（未处理为BIOES标注）取出文件夹内规定数量文件  合并为一个大文件
def mergeFiles(file_dir, n, output_dir, k):
    s = ""
    i = 0
    fileName_list = os.listdir(file_dir)
    for fileName in fileName_list:
        i += 1
        filePath = os.path.join(file_dir, fileName)

        with open(filePath, 'r', encoding='utf-8') as f:
            temp = f.read()
            if not temp.isspace():
                s += temp + "\n"
        if i % n == 0:
            newfile = os.path.join(output_dir, "{}.txt".format(i // n + k))
            with open(newfile, "w", encoding='utf-8') as f:
                f.write(s)
            s = ""
        if i == len(fileName_list):
            newfile = os.path.join(output_dir, "{}.txt".format(i // n + 1 + k))
            with open(newfile, "w", encoding='utf-8') as f:
                f.write(s)
            s = ""
    print(i, "files have been merged. Filedir: ", file_dir)
    return i // n + 1


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


def read_train_data(traindata_path):
    '''
    读取标注好的BIOES文件，转换为模型所需的列表格式   TODO：加入 字->id 的映射
    :param traindata_path:
    :return:
    '''
    data = []
    with io.open(traindata_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    chars, tags, tag_ids = [], [], []
    for line in lines:
        if line != '\n':
            # [char, label] = line.strip().split()
            try:
                char = ''.join(line).strip().split()[0]
                chars.append(char)
                tag = ''.join(line).strip().split()[1]
                tags.append(tag)
                tag_id = tag2id[tag]
                tag_ids.append(tag_id)
            except Exception:
                print(line)
        else:
            if len(chars) < 1 or len(tags) < 1:
                continue
            data.append((chars, tags, tag_ids))
            chars, tag, tag_ids = [], [], []
    print(traindata_path, ':', len(data))
    # print(data)
    return data

# 人民日报语料库专用 转换BIOES：
def text2BIOES(path, outpath):
    '''
    1词1行的文本转换成BIOES标注的
    :param path:
    :return:
    '''
    with open(path, 'r', encoding='utf-8') as f:
        # lines = f.readlines()
        text = f.read()
        text.replace('。”', '。/w\n”/w\n')

    lines = text.split('\n')
    BIOES_content = ""
    for i, line in enumerate(lines):
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
                # print(e)
                print('=========this line is:', i, ':', line)
                if len(line) == 1:
                    BIOES_content += line + '\tO' + '\n'
                else:
                    try:
                        word, _, tag = line.split('/')
                        BIOES_content += '/\tO' + '\n'
                    except Exception:
                        for char in line:
                            BIOES_content += char + '\tO' + '\n'

        else:
            BIOES_content += '\n'
    with open(outpath, 'w', encoding='utf-8') as f:
        f.write(BIOES_content)
    print('done!->', path)

# 人民日报语料库专用函数 合并多个BIOES文件为一个
def merge_BIOES_files(dir, output_path):
    content_merged = ''
    files = os.listdir(dir)
    for file in files:
        file_path = os.path.join(dir, file)
        try:
            with open(file_path,'r',encoding='utf-8') as f:
                content = f.read()
                content = re.sub('\n{3,}', '\n\n', content)
        except Exception as e:
            print(e)
            print('!!!!!!!!!Error in ', file_path)
        content_merged += (content)
    with open(output_path,'w',encoding='utf-8') as f:
        f.write(content_merged)

# 根据语料 建立词表映射（字<->id）[语料是已经转换为BIOES的文件]
def vocab_build(corpus_dir, vocab_path, min_count):
    '''

    :param corpus_path: 语料路径
    :param vocab_path:  词表路径
    :param min_count:   最小字频（生僻字）
    :return:            字表（字<->id）
    '''
    files = os.listdir(corpus_dir)
    word2id = {}
    for file in files:
        corpus_path = os.path.join(corpus_dir, file)
        with io.open(corpus_path,'r',encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line is not '':
                word, _ = line.split('\t')
                if word.isdigit():
                    word = '<NUM>'
                elif ('\u0041' <= word <='\u005a') or ('\u0061' <= word <='\u007a'):
                    word = '<ENG>'
                if word not in word2id:
                    word2id[word] = [len(word2id)+1, 1]
                else:
                    word2id[word][1] += 1
        print('done!-------->',corpus_path)
    # low_freq_words = []
    # for word, [word_id, word_freq] in word2id.items():
    #     if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
    #         low_freq_words.append(word)
    # for word in low_freq_words:
    #     del word2id[word]
    #
    # new_id = 1
    # for word in word2id.keys():
    #     word2id[word] = new_id
    #     new_id += 1
    # word2id['<UNK>'] = new_id
    # word2id['<PAD>'] = 0

    print(len(word2id))
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)

# 原BIOES中的PER、LOC、ORG转换成 Person、Loc、Org
def BIOES_tag_trans(path,outputpath):
    with open(path,'r',encoding='utf-8') as f:
        content = f.read()
    newcontent = content.replace('LOC','Loc').replace('ORG','Org').replace('PER','Person')
    with open(outputpath, 'w',encoding='utf-8') as f:
        f.write(newcontent)


def vocab_trans(oVocab_path,out_path):
    with open(oVocab_path,'rb') as fr:
        dict = pickle.load(fr)
    newdict = {k:v[0] for k,v in dict.items()}
    print(newdict)
    with open(out_path,'wb') as fw:
        pickle.dump(newdict,fw)

# d读取词表 返回字典（字->id）
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

# BIO格式标注数据 转换为BIOES格式：
def BIO2BIOES(path, outpath):
    '''
    :param path:
    :return:
    '''
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    BIOES_content = ""
    for i, line in enumerate(lines):
        print(i)
        line = line.strip()
        if line is not '':
            word, tag = line.split()
            # 单独的 B 改成 S
            if tag[0] is 'B':
                if i+1 < len(lines) and lines[i+1].strip() is not '':
                    _, next_tag = lines[i+1].split()
                    if next_tag[0] is not 'I':
                        tag = 'S' + tag[1:]
                else:
                    tag = 'S' + tag[1:]
            # 最后一个 I 改成 E
            elif tag[0] is 'I':
                if i + 1 < len(lines) and lines[i + 1].strip() is not '':
                    _, next_tag = lines[i + 1].split()
                    if next_tag[0] is 'I':
                        continue
                tag = 'E' + tag[1:]

            BIOES_content += word + '\t' + tag +'\n'

        else:
            BIOES_content += '\n'

    with open(outpath, 'w', encoding='utf-8') as f:
        f.write(BIOES_content)
    print('done!->', path)


if __name__ == '__main__':
    # indir = r'F:\zzd\毕业论文\论文代码\DataSets\2014人民日报\BIO'
    # outpath = r'F:\zzd\毕业论文\论文代码\NER\data\人民日报'
    # merge_BIOES_files(indir, outpath)


    # 1词1行转BIOES
    # file_dir = os.listdir(inputdir)
    # outfiles = os.listdir(outdir)
    # for file_name in file_dir:
    #     if file_name in outfiles:
    #         continue
    #     inputpath = os.path.join(inputdir, file_name)
    #     outpath = os.path.join(outdir, file_name)
    #     text2BIOES(inputpath, outpath)

    # dir = r'F:\zzd\毕业论文\论文代码\NER\data'
    vocab_cout_path = r'F:\zzd\毕业论文\论文代码\NER\vocab\vocab_count.pkl'
    new_outpath = r'F:\zzd\毕业论文\论文代码\NER\vocab\vocab.pkl'


