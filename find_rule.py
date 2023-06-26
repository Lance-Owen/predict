import matplotlib.pyplot as plt
import pandas as pd
from tools import *
df = pd.read_csv('lishui.csv',encoding='gbk')
df =df[df['k1'] !=0 ]
df =df[df['ZBKZJ'] !=0 ]
# df = df.drop_duplicates(subset=['ZBKZJ','k1','kbjj'])
# df['Date'] = df['网址'].apply(get_date)

# df.to_csv('lishui.csv',encoding='utf8',index=False)
df = df.sort_values(by=['ZBKZJ'])

df1 = df
plt.scatter(df1['ZBKZJ'],df1['k1'])
plt.scatter(df1['k1'],df1['ZBKZJ'])

df2 = df
plt.scatter(df2['ZBKZJ'],df2['k1'])
plt.scatter(df2['k1'],df2['ZBKZJ'])

plt.rcParams['font.sans-serif']=['SimHei']
plt.hist(df2['k1'], bins=30, density=True, alpha=0.6, color='g')
plt.hist(df2['ZBKZJ'], bins=3000, density=True, alpha=0.6, color='g')
plt.xlabel('X轴标签')
plt.ylabel('Y轴标签')
plt.title('数据的连续分布情况')
plt.show()


import random
import time

data_list = []
for i in range(0,1000):
    a = random.randint(6,19)
    time.sleep(random.random())
    b = random.randint(0,9)
    data_list.append(a+b*0.1)
plt.hist(data_list, bins=30, density=True, alpha=0.6, color='g')
plt.xlabel('X轴标签')
plt.ylabel('Y轴标签')
plt.title('数据的连续分布情况')
plt.show()
