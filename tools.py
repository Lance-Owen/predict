import re
import datetime
import pandas as pd

def get_date(str_time:str):
    # str_time = 'http://lssggzy.lishui.gov.cn/art/2020/10/16/art_1229661956_138130.html'
    time = re.findall('(\d{4})\/(\d{1,2})\/(\d{1,2})',str_time)[0]
    s = "-".join(time)
    return s
    # datetime.datetime.strptime(s, '%Y-%m-%d')

def zt(d2):
    t3 = d2.groupby(['k1'])['ZBKZJ'].count() \
        .reset_index(name='count') \
        .sort_values(['count'], ascending=False) \
        .head(5)
    # print(t3)
    return t3.max()

def lishui_data():
    df = pd.read_csv('lishui.csv', encoding='gbk')
    df['k2'] = 0
    # df = df[['ZBKZJ','k1','k2']]
    df =df[df['k1'] !=0 ]
    df =df[df['ZBKZJ'] !=0 ]

    return df