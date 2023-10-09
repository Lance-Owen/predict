import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tools import *
import datetime
df = pd.read_csv('lishui.csv',encoding='gbk')
df['Date'] = df['网址'].apply(get_date)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['zbkzj'])
# df = df.sort_values(by=['Date'])
df = df[df['Date'] > '2023-01-01']

values = []
mean_values = []

length = 10000
width = 5

df = df[(df['下浮率'] > 6) & (df['下浮率'] < 12)]
print(len(df))

values = df['下浮率'].values.tolist()
# for i in range(length):
#     values.append(random.randrange(start=60, stop=120, step=1) / 10)
#     # mean_values.append(sum(values)/len(values))

for i in range(width, len(values)-width):
    mean_values.append(sum(values[i - width:i+width+1]) / (width*2+1))
    # x = values[i-3:i]
    # x = sorted(x,reverse=True)
    # mean_values.append(sum(np.array(x)*np.array([0.2,0.5,0.3])))
plt.rcParams['font.sans-serif']=['SimHei']
plt.plot(range(len(values)), values, marker='o',label='原始数据')
plt.plot(range(width,width+len(mean_values)), mean_values, marker='o',label='测试数据')
plt.plot(range(len(values)),[9]*len(values))
plt.legend()
plt.title(f'{datetime.datetime.now()}')
plt.show()
print(df['下浮率'].values.mean())

print(len(df[df['下浮率']>=9]))
print(len(df[df['下浮率']<9]))
print(mean_values)

# 规定一个宽度，在数据两端取数据，计算均值
# 观察到均值的前后浮动大小会在一个值的范围内变化，但是可以一直增长或降低
