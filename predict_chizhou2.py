import matplotlib.pyplot as plt
import pandas as pd
import re

plt.rcParams['font.sans-serif'] = ['SimHei']

df = pd.read_excel('池州市项目预测0801.xlsx')
df['开标时间'] = pd.to_datetime(df['开标时间'], format="%Y-%d-%m %H:%M:%S")
df = df.sort_values(by=['开标时间'])
print(df.columns)
df1 = df[['zbkzj', 'k1', 'k2', '下浮率', '开标时间', 'project_type', '取值范围']]
df1[['zbkzj', 'k1', 'k2', '下浮率']].fillna(0, inplace=True)
df1['project_type'].fillna('', inplace=True)


def predict_rate(kzj, project_type, df1):
    df1 = df1[
        (df1['zbkzj'] >= 0.9 * kzj) & (df1['zbkzj'] <= 1.1 * kzj) & (df1['project_type'].str.contains(project_type))]
    if len(df1)<=1:
        return '无值'
    return df1['下浮率'].mean()


kzj = 41020100
project_type = '市政'
print(predict_rate(kzj, project_type, df1))

# df['project_type'].fillna('', inplace=True)
# # df = df[df['project_type'].str.contains('市政公用工程施工总承包|建筑工程施工总承包|房屋建筑工程施工总承包|水利水电工程施工总承包|公路工程施工总承包')]
# df1['predict_rate'] = df1.apply(lambda row:predict_rate(row['zbkzj'],row['project_type'],df),axis=1)
# df1.to_csv('池州市1.csv')



# 计算项目价格相差上下10%，资质相同的项目，计算出均值作为预测值。