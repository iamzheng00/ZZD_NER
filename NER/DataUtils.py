import io
import json
import os, re
import Data_processing

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


# 取出文件夹内规定数量文件  合并为一个大文件
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


def maxlen(sentences):
    return max([len(s) for s in sentences])


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
    BIOES标注好的文本 读取后转换为模型所需列表
    :param traindata_path: BIOES标注好的文本路径
    :return: data=[[sentence],[sentence],....]
            sentence=[[chars],[tags]]
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



# for NER
if __name__ =='__main__':
    path = r'F:\zzd\毕业论文\论文代码\DataSets\2014人民日报\BIO\1.txt'
    data = read_train_data(path)
    A = data
