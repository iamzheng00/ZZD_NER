import io
import os
import pickle
import re

tag2id = {'O': 0,
          'S-Person': 1, 'B-Person': 2, 'I-Person': 3, 'E-Person': 4,
          'S-ORG': 5, 'B-ORG': 6, 'I-ORG': 7, 'E-ORG': 8,
          'S-LOC': 9, 'B-LOC': 10, 'I-LOC': 11, 'E-LOC': 12,
          'S-Time': 13, 'B-Time': 14, 'I-Time': 15, 'E-Time': 16,
          }

RMRB_tag = {
    "nr": "Person",
    "ns": "Loc",
    "nt": "Org",
    "t": "Time"
}


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

# 合并多个BIOES文件为一个
def merge_BIOES_files(path, output_file_name):
    content_merged = ''
    files = os.listdir(path)
    for file in files:
        file_path = os.path.join(path, file)
        try:
            with io.open(file_path) as f:
                content = f.read()
                content = re.sub('\n{3,}', '\n\n', content)
        except Exception as e:
            print(e)
            print('!!!!!!!!!Error in ', file_path)
        content_merged.join(content)
    output_path = os.path.join(path, output_file_name)
    with io.open(output_path) as f:
        f.write(content_merged)

# 根据语料 建立词表映射（字<->id）[语料是已经转换为BIOES的文件]
def vocab_build(corpus_path, vocab_path, min_count):
    '''

    :param corpus_path: 语料路径
    :param vocab_path:  词表路径
    :param min_count:   最小字频（生僻字）
    :return:            字表（字<->id）
    '''
    with io.open(corpus_path,'r',encoding='utf-8') as f:
        lines = f.readlines()
    word2id = {}
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

if __name__ == '__main__':
    indir = r'F:\zzd\毕业论文\论文代码\DataSets\2014人民日报\BIO\1.txt'
    outdir = r'F:\zzd\毕业论文\论文代码\DataSets\2014人民日报\vocab_test.pkl'

    # vocab_build(indir,outdir,5)
    w2id = read_vocab(outdir)
    print(type(w2id))
    print(w2id['我'])

    # 1词1行转BIOES
    # file_dir = os.listdir(inputdir)
    # outfiles = os.listdir(outdir)
    # for file_name in file_dir:
    #     if file_name in outfiles:
    #         continue
    #     inputpath = os.path.join(inputdir, file_name)
    #     outpath = os.path.join(outdir, file_name)
    #     text2BIOES(inputpath, outpath)
