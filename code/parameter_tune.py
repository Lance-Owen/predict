import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']

from tools import *

df = pd.read_excel(r"C:\Users\pc\Desktop\池州市项目预测.xlsx")
df = df.fillna("")
df = df[df['group']!='']
# df = df.sort_values(by=['zbkzj'])
# df = df[(df['zbkzj'] <= 7*10**7)]
# df = df[df['project_type'] == '建筑工程施工']
# df = df[df['project_type'] == '市政公用工程']
# df['publish_time'] = pd.to_datetime(df['开标时间'])
# df = df.sort_values(by=['开标时间'])
# plt.plot(data['publish_time'], data[['m', 'x1', 'x2', 'x3', 'B']], label=['m', 'x1', 'x2', 'x3', 'B'])
plt.plot(df['开标时间'], df['group'])
plt.title('group走势')
# plt.title('市政公用工程按kzj下浮率走势')

# plt.legend()
plt.show()
