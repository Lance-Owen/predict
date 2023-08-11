import matplotlib.pyplot as plt
import pandas as pd
import re

plt.rcParams['font.sans-serif'] = ['SimHei']

df = pd.read_csv('淮南市数据.csv')
# df['publish_time'] = pd.to_datetime(df['publish_time'], format="%Y-%d-%m %H:%M:%S")
df = df.sort_values(by=['publish_time'])
print(df.columns)
df['下浮率'] = 100-100*df['kbjj']/df['zbkzj']

df1 = df[['zbkzj', 'k1', 'k2', '下浮率', 'publish_time', 'project_type']]
df1[['zbkzj', 'k1', 'k2', '下浮率']].fillna(0, inplace=True)
df1['project_type'].fillna('', inplace=True)


def predict_rate(kzj, project_type, df1):
    df1 = df1[
        (df1['zbkzj'] >= 0.9 * kzj) & (df1['zbkzj'] <= 1.1 * kzj) & (df1['project_type'].str.contains(project_type))]
    if len(df1)<=1:
        return '无值'
    return df1['下浮率'].mean()


kzj = 74153900
project_type = '市政'
print(predict_rate(kzj, project_type, df1))
print(set(df1['project_type'].values.tolist()))