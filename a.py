import random
import pandas as pd





# 只有一个数字符合均匀分布
def avg(d2,my_list):
     t3 = d2.groupby('k1')['Date'].count()\
        .reset_index(name='count') \
        .sort_values(['count'], ascending=False) \
        .head(5)
     return t3.min()['k1']




if __name__ == '__main__':
    # 定义随机数
    my_list = ['0.97','0.98','0.99','1','1.01','1.02']
    my_list2 = ['0.97', '0.98', '0.99', '1', '1.01', '1.02']
    df = pd.read_csv('20230619.csv', encoding='utf-8')
    d2 = df[['Date', 'k1']]
    length = 9
    result = d2.sample(n=length)
    print(avg(result,my_list))
