import datetime
import random
import requests
from bs4 import BeautifulSoup
import bs4
import os
import time
import jsonlines
import json
from selenium import webdriver

import io
import sys


#
# class FeatureStorey(object):
#     def __init__(self, featurestory_title, news_date, url, news_content):
#         self.FeatureStorey_title = featurestory_title
#         self.FeatureStorey_date = news_date
#         self.FeatureStorey_url = url
#         self.FeatureStorey_content = news_content

# post 获取html文本
def post_html(url, pagestart):
    headers1 = {
        'Host': 'sou.chinanews.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:71.0) Gecko/20100101 Firefox/71.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
        'Accept-Encoding': 'gzip, deflate',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Content-Length': 29,
        'Origin': 'http://sou.chinanews.com',
        'Connection': 'keep-alive',
        'Referer': 'http://sou.chinanews.com/search.do',
        'Cookie': 'cnsuuid=1432931f-1cef-b916-57ef-4f8d5a06a98414825.113891283227_1578547697387; __'
                  'jsluid_h=26f22ca3655a43605b464b1f4ce4d008; zycna=iO8F8n2pDXQBAasrvIu/kmUP; '
                  'JSESSIONID=aaav_9notfMU74a55dk_w; Hm_lvt_0da10fbf73cda14a786cd75b91f6beab=1578550366,1578570043; Hm_lpvt_0da10fbf73cda14a786cd75b91f6beab=1578570043; __jsl_clearance=1578573762.959|0|jHNJ%2F84E%2FjsBxBOYzHG0OQHxhUI%3D',
        'Upgrade-Insecure-Requests': 1
    }
    headers2 = {
        'Host': 'www.xinhuanet.com',
        'Connection': 'keep-alive',
        # 'Content-Length': 29,
        'Cache-Control': 'max-age=0',
        'Upgrade-Insecure-Requests': 1,
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/79.0.3945.117 Safari/537.36',
        # 'Origin': 'http://sou.chinanews.com',
        # 'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': ' text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        # 'Referer': 'http://sou.chinanews.com/search.do',
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': ' zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
        'Cookie': 'wdcid=64207473d35b9c56; wdlast=1578566647'
    }
    params = {
        'q': '%E6%AD%BC%E5%87%BB%E6%9C%BA',
        # 'ps': 10,
        # 'start': pagestart,
        # 'type': '',
        # 'sort': 'pubtime',
        # 'time_scope': 0,
        # 'channel': 'all',
        # 'adv': 1,
        # 'day1': '',
        # 'day2': '',
        # 'field': '',
        # 'creator': '',
    }
    # post 获取Ajax 局部更新的html文本
    try:
        resp = requests.post(url, headers=headers1, data=params)
    except:
        print('request failed')
        return None
    if resp.status_code == 200:
        return resp.text
    return None


# get 获取html文本
def get_html(url):
    headers = {
        'Host': 'news.baidu.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:72.0) Gecko/20100101 Firefox/72.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': ' https://www.baidu.com/link?url=dnoX1K64M45SBGpTLFBOYvmZP7zBG5-5T1nct4mVhpzE-hX'
                   'Z-pveCghXOQKtWt1E&wd=&eqid=cfde344a000054eb000000065e170993',
        'Connection': 'keep-alive',
        # 'Cookie': 'wdcid=64207473d35b9c56; wdlast=1578566647',
        'Upgrade-Insecure-Requests': 1,
        'Cache-Control': 'max-age=0'

    }
    try:
        resp = requests.get(url, headers=headers)
    except:
        print('request failed')
        return None
    print('resp.status_code = ', resp.status_code)
    if resp.status_code == 200:
        return resp.text
    return None


def get_inf_list(html):
    t = 0
    soup = BeautifulSoup(str(html), 'lxml')
    one_page_newslist = soup.find('div', id='news_list').select('table')
    inf_list = []
    for one_newsinf in one_page_newslist:
        title = one_newsinf.find('li', class_='news_title').a.get_text().replace('\n', '')
        if (type(one_newsinf.find('li', class_='news_title').a) is 'NoneType'):
            continue
        inf_url = one_newsinf.find('li', class_='news_title').a['href']
        date_pub = one_newsinf.find('li', class_='news_other').get_text().split()[-2].replace('-', '')

        content = get_inf_content(inf_url)
        if (content is None):
            continue
        date_now = datetime.datetime.now().strftime('%Y%m%d')
        one_inf = {
            'Title': title,
            'URL': inf_url,
            'Details': content,
            'Source': 'News',
            'Field': 'airforce',
            'Label': 'Weapon',
            'Date_pub': date_pub,
            'Date_crawl': date_now,
            'More': 'null',
        }
        inf_list.append(one_inf)
        t=t+1
        print(t, '  is ready ')

    return inf_list


def get_inf_content(url):
    browser0 = webdriver.Chrome()
    browser0.get(url)

    content_page = browser0.page_source

    soup = BeautifulSoup(content_page, 'lxml')
    browser0.close()

    tmp = soup.find('div', class_='left_zw')
    if (tmp):
        content_text = tmp.get_text().split('【')[0].replace('\n', '').replace(' ', '')
        return content_text
    else:
        return None


def write_tofile(inf_list, foldername, filename):
    global n
    t = 0 # 计数

    path = 'D:/project2/数据爬取/2020_数据爬取/Spider/' + foldername
    filepath = path + '/' + filename + '.json'

    if not os.path.exists(path):
        os.mkdir(path)
        print(os.path.abspath(path))
    for article_info in inf_list:
        with jsonlines.open(filepath, mode='a') as f:
            f.write(article_info)
            # print(os.path.abspath(filepath))
        t = t + 1
        print(t , '  is written to json')





def printNewsList(newsList):
    counter = 1
    for x in newsList:
        print( '*', counter, '*')
        counter += 1
        print( 'Title:', x['Title'])
        print( 'URL:', x['URL'])
        # print( 'Details:', x['Details'])
        print( 'Source:', x['Source'])
        print( 'Field:', x['Field'])
        print( 'Label:', x['Label'])
        print( 'Date_pub:', x['Date_pub'])
        print( 'Date_crawl:', x['Date_crawl'])
        print( 'More:', x['More'])




USER_AGENTS = [
            "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)",
            "Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)",
            "Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)",
            "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)",
            "Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6",
            "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1",
            "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0",
            "Mozilla/5.0 (X11; Linux i686; U;) Gecko/20070322 Kazehakase/0.4.5",
            "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3192.0 Safari/537.36Name"
            ]


# res = requests.get('http://sou.chinanews.com/search.do?q=%E6%AD%BC%E5%87%BB%E6%9C%BA&start=10')





sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

n= 0

for Page in range(0, 10):
    print('======', 'page:', Page + 1, '========')
    # 模拟浏览器打开
    browser = webdriver.Chrome()
    keyword = '格斗导弹'
    URL = 'http://sou.chinanews.com/search.do?q=' + keyword + '&start=' + str(Page * 10)
    browser.get(URL)
    #时延后获取网页源码
    time.sleep(3)
    HTML = browser.page_source
    browser.close()

    #获取当前页的所有新闻列表信息
    newsList = get_inf_list(HTML)

    #打印当前页的每条新闻信息
    printNewsList(newsList)

    #写入json文件
    foldername = 'airfoce/news'
    write_tofile(newsList, foldername=foldername, filename=keyword)
    if(n > 50):
            break




