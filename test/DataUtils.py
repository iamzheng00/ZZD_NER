import io
import json
import os, re
import DataCollecting


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


def textReplace(text):
    """
    去除文中[ ]复核标签内的多余标签，只取外层标签
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


def maxlen(batch):
    pass  # TODO


def tag_change(tag):
	"""
	TODO
	原标注转换为需要的标注
	:param tag:
	:return:
	"""
	return tag


def read_corpus(corpus_path):
    '''
    :param corpus_path: corpus是BIOES标注好的文本
    :return: data=[[sentence],[sentence],....]
            sentence=[[chars],[tags]]
    '''
    data = []
    with io.open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    chars, tags = [], []
    for line in lines:
        if line != '\n':
            # [char, label] = line.strip().split()
            try:
                char = ''.join(line).strip().split()[0]
                chars.append(char)
                tag = ''.join(line).strip().split()[1]
                tag = tag_change(tag)
                tags.append(tag)
            except Exception:
                print(line)
        else:
            if len(chars)<1 or len(tags)<1:
                continue
            data.append((chars, tags))
            chars, tags = [], []
    print(corpus_path, ':', len(data))
    # print(data)
    return data


path = 'F:/zzd/毕业论文/论文代码/DataSets/2014人民日报/testout.txt'

data = read_corpus(path)
A = data