import random
import pandas as pd



#两个数字符合正态分布
def zt(d2):
    t3 = d2.groupby(['k1','k2'])['ZBKZJ'].count() \
        .reset_index(name='count') \
        .sort_values(['count'], ascending=False) \
        .head(5)
    # print(t3)
    return t3.max()

def lishui_data():
    df = pd.read_csv('lishui.csv', encoding='gbk')
    df['k2'] = 0
    df = df[['ZBKZJ','k1','k2']]
    df =df[df['k1'] !=0 ]
    df =df[df['ZBKZJ'] !=0 ]

    return df


if __name__ == '__main__':
    #length=(len1+len2)*1.5
    df = pd.read_csv('20230605.csv', encoding='utf-8')

    df = lishui_data()
    d2 = df[['k1','k2','ZBKZJ']]
    d2 =d2[d2['k1'] !=0 ]
    length = 50
    result = d2.sample(n=length)
    print(zt(result)['k1'])






