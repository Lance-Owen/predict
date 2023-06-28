import re
import datetime
import pandas as pd
import random
import chardet


def get_date(str_time:str):
    # str_time = 'http://lssggzy.lishui.gov.cn/art/2020/10/16/art_1229661956_138130.html'
    time = re.findall('(\d{4})\/(\d{1,2})\/(\d{1,2})',str_time)[0]
    s = "-".join(time)
    return s

def read_file(file_path):
    for ed in ['gbk','utf-8']:
        try:
            df = pd.read_csv(file_path, encoding=ed)
            break
        except:
            pass
    return df

def zt(d2):
    t3 = d2.groupby(['k1','k2'])['zbkzj'].count() \
        .reset_index(name='count') \
        .sort_values(['count'], ascending=False) \
        .head(5)
    # print(t3)
    return t3.max()

def rule1(df):

    length = int(len(df)*0.1)
    df = df.sample(n=length)
    k1_list = df['下浮率'].values.tolist()
    if k1_list.count(max(k1_list,key=k1_list.count))==1:
        return max(k1_list,key=k1_list.count)
    else:
        return max(k1_list)

def rule2(df):

    length = int(len(df)*0.3)
    df = df.sample(n=length)
    k1_list = df['下浮率'].values.tolist()
    k1_dict = {}
    for key in set(k1_list):
        k1_dict[key] = k1_list.count(key)
    return_k1 = 0
    for k,v in k1_dict.items():
        return_k1 += round(int(k)*int(v)/len(k1_list),2)
    # print(return_k1)
    return return_k1

def rule3(df):
    return round(df['下浮率'].mean()+random.uniform(-0.6,0.6),2)

def lishui_data(file_path):
    df = read_file(file_path)
    df['k1'] = 0
    df['zbkzj'] = df['zbkzj'].astype(float)
    df['kbjj'] = df['kbjj'].astype(float)

    df =df[df['zbkzj'] !=0 ]
    # df['Date'] = pd.to_datetime(df['Date'])

    df['整数位'] = df['k1'].apply(lambda s:str(1000-int(1000*s))[:-1])
    df['小数位'] = df['k1'].apply(lambda s:str(1000-int(1000*s))[-1])
    df['下浮率'] = df['k1'].apply(lambda s:str(round(100-100*s,2)))
    # df['下浮率'] = df['下浮率'].astype(float)
    df['下浮率'] = round(100-100*df['kbjj']/df['zbkzj'],2)
    df = df[df['下浮率']>6]
    df = df[df['下浮率']<12]
    return df

def luan_data():
    df = read_file('luan.csv')
    df['k1'] = 0
    df['k2'] = 0

    df = df[df['zbkzj']>df["kbjj"]]
    df['下浮率'] = 100-round(df['kbjj']/df['zbkzj']*100,2)
    df = df[df['下浮率']<30]
    return df

def huainan_data():
    # df = read_file("lishui.csv")
    df = pd.read_csv('huainan.csv',encoding='utf-8')

    # df = df[df['k1']>0]
    # df = df[df['k2']>0]
    # df = df[df['k1']<1]
    # df = df[df['k2']<1]

    df = df[df['zbkzj']>df["kbjj"]]
    df['下浮率'] = 100-round(df['kbjj']/df['zbkzj']*100,2)
    df = df[df['下浮率']<20]

    return df
