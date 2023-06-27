import matplotlib.pyplot as plt
import pandas as pd
from decimal import Decimal
from tools import *
df = lishui_data()


df['下浮率'] = df['k1'].apply(lambda s:str(round(100-100*s,2)))

df['下浮率'] = df['下浮率'].astype('Float64')

df = df[df['下浮率']>6]
df = df[df['下浮率']<12]
df['整数位'] = df['k1'].apply(lambda s:str(1000-int(1000*s))[:-1])
df['小数位'] = df['k1'].apply(lambda s:str(1000-int(1000*s))[-1])


plt.scatter(range(len(df)),df['k1'])



# df = df[['ZBKZJ','k1','k2']]
# 根据时间分割数据集合
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['k1'])
plt.plot(range(len(df)),df['k1'])
# df = df.drop_duplicates(subset=['ZBKZJ','k1','kbjj'])
# df['Date'] = df['网址'].apply(get_date)

df.to_csv('测试数据.csv',encoding='utf8',index=False)
df = df.sort_values(by=['ZBKZJ'])



plt.rcParams['font.sans-serif']=['SimHei']
plt.hist(df['下浮率'], bins=30, density=True, alpha=0.6, color='g')
# plt.hist(df['ZBKZJ'], bins=3000, density=True, alpha=0.6, color='g')
plt.xlabel('X轴标签')
plt.ylabel('Y轴标签')
plt.title('数据的连续分布情况')
plt.show()


# 随机历史数据，生成预测数据
import random
import time
from tools import *
import matplotlib.pyplot as plt



data = lishui_data()

data_list = []
for i in range(1000):
    length = 6
    result = data.sample(n=length)
    # print(zt(result)['k1'])
    data_list.append(zt(result)['k1'])

plt.hist(data_list, density=True, alpha=0.6, color='g')
plt.xlabel('X轴标签')
plt.ylabel('Y轴标签')
plt.title('数据的连续分布情况')
plt.show()




chatgpt_k1 = [8.3, 7.6, 9.1, 8.8, 7.9, 8.5, 7.2, 9.0, 8.2, 8.6, 7.7, 8.4, 8.1, 7.8, 8.0, 7.5, 8.7, 7.4, 8.9, 7.3, 8.8, 7.6, 9.2, 8.3, 7.9, 9.1, 8.5, 7.7, 9.0, 8.6, 7.8, 8.7, 7.5, 8.9, 7.4, 8.8, 7.6, 9.3, 8.4, 7.9, 9.2, 8.5, 7.7, 9.1, 8.6, 7.8, 9.0, 8.7, 7.5, 9.2]

