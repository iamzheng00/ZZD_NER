# encoding='utf-8'
import os
import re
from DataUtils import mergeFiles
from DataUtils import textReplace

# dir1 = "F:/zzd/毕业论文/数据集/2014人民日报（原）"
# dir2_list = os.listdir(dir1)
# outputdir = "F:/zzd/毕业论文/数据集/te"
# k = 0
# for dir2 in dir2_list:
# 	file_dir = os.path.join(dir1, dir2)
# 	t = mergeFiles(file_dir, 100, outputdir, k)
# 	k = t + k

# path = "../DataSets/2014人民日报/1文本100篇"
# filelist = os.listdir(path)
# for file in filelist:
#     filepath = os.path.join(path, file)
#     with open(filepath,'r+',encoding='utf-8') as f:
#         text = f.read()
#         q = textReplace(text)
#         f.seek(0)
#         f.truncate()
#         f.write(q)
#     print("done! filepath: ",filepath)
"""
转换为BIOES
"""


def isTagNeed(tag):
    """
    TODO
    是否保留
    :param tag:
    :return:
    """
    return True


def tag_change(tag):
    """
    TODO
    原标注转换为需要的标注
    :param tag:
    :return:
    """
    return tag


path = '../../DataSets/2014人民日报/test.txt'
outpath = '../DataSets/2014人民日报/testout.txt'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()
l = text.split('\n')
q = ""
lineno = 0
for i in l:
    lineno += 1
    print(lineno)
    if i is not '':
        try:
            word, tag = i.split('/')
            if len(word) == 1:
                q += word + '\tS-' + tag + '\n'
            elif len(word) > 1:
                tag = tag_change(tag)  # TODO
                q += word[0] + '\tB-' + tag + '\n'
                for char in word[1:-1]:
                    q += char + '\tI-' + tag + '\n'
                q += word[-1] + '\tE-' + tag + '\n'
        except Exception as e:
            print(e)
            word, _, tag = i.split('/')
            q += '/\tS-' + tag + '\n'
    else:
        q += '\n'

with open(outpath, 'w', encoding='utf-8') as f:
    f.write(q)
