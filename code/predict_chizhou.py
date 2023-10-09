import re

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']

from tools import *

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor as XGBR


def train_model_predict(df):
    # d2 = df[['zbkzj','k1','k2','kbjj']]

    # 提取数据
    d1 = df.dropna().reset_index(drop=True)
    X = d1.drop(columns=['下浮率'])
    y = d1['下浮率']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=7)
    poly = PolynomialFeatures(degree=1)

    x_train = poly.fit_transform(X_train.values)
    x_test = poly.fit_transform(X_test)

    # 线性回归
    la = Lasso(alpha=0.1, max_iter=100000)
    la.fit(x_train, y_train)
    # print(f'线性回归训练集得分：{round(la.score(x_train,y_train),2)}')
    # print(f'线性回归测试集得分：{round(la.score(x_test,y_test),2)}')

    # 随机森林回归
    rf = RandomForestRegressor(n_jobs=-1, max_depth=None, min_samples_leaf=1, min_samples_split=13, n_estimators=10)
    rf.fit(x_train, y_train)
    # print(f'随机森林回归训练集得分：{round(rf.score(x_train,y_train),2)}')
    # print(f'随机森林回归测试集得分：{round(rf.score(x_test,y_test),2)}')

    # 决策树回归
    dt = DecisionTreeRegressor(max_depth=6)
    dt.fit(x_train, y_train)
    # print(f'决策树回归训练集得分：{round(dt.score(x_train,y_train),2)}')
    # print(f'决策树回归测试集得分：{round(dt.score(x_test,y_test),2)}')

    # K近邻回归
    kn = KNeighborsRegressor(n_neighbors=3, n_jobs=-1)
    kn.fit(x_train, y_train)
    # print(f'k近邻回归测试集得分：{round(kn.score(x_test,y_test),2)}')

    # XGBbost
    xgb = XGBR(n_estimators=10)
    xgb.fit(x_train, y_train)
    # print(f"xgbbost测试集得分：{round(xgb.score(x_test,y_test),2)}") #你能想出这里应该返回什么模型评估指标么?

    return la, rf, dt, kn, xgb


def record_predict(df_train, predict_value):
    # predict_k1 = zt(df.sample(n=8))['k1']
    # predict_k2 = zt(df.sample(n=8))['k2']

    la, rf, dt, kn, xgb = train_model_predict(df_train)

    poly = PolynomialFeatures(degree=1)
    apply = np.array(predict_value).reshape(1, -1)
    poly_apply = poly.fit_transform(apply)

    predict_list = [la.predict(poly_apply).tolist()[0], rf.predict(poly_apply).tolist()[0],
                    dt.predict(poly_apply).tolist()[0], kn.predict(poly_apply).tolist()[0]]

    res = (sum(predict_list) - max(predict_list) - min(predict_list)) / 2

    xgb_mean = sum([rf.predict(poly_apply), kn.predict(poly_apply), xgb.predict(poly_apply)]) / 3

    # print(xgb.predict(poly_apply).tolist()[0], xgb_mean[0], res)
    return la.predict(poly_apply).tolist()[0], rf.predict(poly_apply).tolist()[0], dt.predict(poly_apply).tolist()[0], \
        kn.predict(poly_apply).tolist()[0], res  # , xgb.predict(poly_apply).tolist()[0],


def predict_rate(df, df_train):
    df[['k1', 'k2', 'zbkzj', 'kbjj']] = df[['k1', 'k2', 'zbkzj', 'kbjj']].fillna(0)
    df = df[['zbkzj', 'kbjj', 'project_type']]
    df = df.fillna('')

    df[['zbkzj']] = df[['zbkzj']].astype(float)
    df[['kbjj']] = df[['kbjj']].astype(float)

    df = df[df['zbkzj'] > df["kbjj"]]
    df['下浮率'] = 100 - round(df['kbjj'] / df['zbkzj'] * 100, 4)
    df = df[df['下浮率'] < 20]

    project_label = {'水利水电工程': 1, '公路工程': 2, '市政公用工程': 3, '建筑工程': 4, '建筑装修装饰工程': 5}
    df['project_label'] = df['project_type'].apply(lambda s: project_label[s] if s in project_label.keys() else 6)

    df['预测下浮率'] = df.apply(lambda x: round(record_predict(df_train, [x['zbkzj'], x['project_label']])[-1], 4),
                                axis=1)

    df['result'] = abs(df['下浮率'] - df['预测下浮率'])
    print('机器学习模型预测结果'.center(100, '*'))
    print(
        f"下浮率误差1%以内命中率：{round(len(df[df['result'] < 1]) / len(df), 2) * 100}%，下浮率命中率：{round(len(df[df['result'] == 0]) / len(df), 2) * 100}%")


# 获取数据，得到处理后的总数据
df_total = chizhou_data()
print("总数据数量", len(df_total))

df_train = df_total[['zbkzj', '下浮率', 'project_label']]

df = pd.read_excel(r"C:\Users\pc\Desktop\池州市项目预测.xlsx")
df = df[df['k1']<=20]
print(df.columns)

# sql = "SELECT id,source_website_address,city,zbkzj,kbjj,k1,k2,project_type,publish_time,suck_factor FROM tender_bid_opening where city='池州市'"
# df = mysql_select_df(sql)


# print(set(df['project_type'].values.tolist()))

# 使用机器学习进行预测，使用控制价和项目类型作为参数
# 只保留控制价下浮率和项目类型
# predict_rate(df, df_train)
# df['预测下浮率'] = df.apply(lambda x: record_predict(df_train, [x['zbkzj'], 3])[-1], axis=1)
# df['result'] = abs(df['下浮率'] - df['预测下浮率'])
# print('机器学习模型预测结果'.center(100, '*'))
# print(f"下浮率误差1%以内命中率：{round(len(df[df['result'] < 1]) / len(df), 2) * 100}%，下浮率命中率：{round(len(df[df['result'] == 0]) / len(df), 2) * 100}%")

# 使用机器学习预测单个数据的下浮率
# print(record_predict(df_train, [9216194.55, 1])[-1])


# # 使用随机规则预测下浮率
df['rule1'] = df['zbkzj'].apply(lambda x: rule1(df_total))
df['rule1_result'] = abs(df['下浮率'] - df['rule1'])
print(len(df[df['rule1_result'] < 1]), len(df[df['rule1_result'] == 0]))

df['rule2'] = df['zbkzj'].apply(lambda x: rule2(df_total, x))
df['rule2_result'] = abs(df['下浮率'] - df['rule2'])
print(len(df[df['rule2_result'] < 1]), len(df[df['rule2_result'] == 0]))
print(
    f"下浮率误差1%以内命中率：{round(len(df[df['rule2_result'] < 1]) / len(df), 2) * 100}%，下浮率命中率：{round(len(df[df['rule2_result'] == 0]) / len(df), 2) * 100}%")

df['rule3'] = df['zbkzj'].apply(lambda x: rule3(df_total, x))
df['rule3_result'] = abs(df['下浮率'] - df['rule3'])
print(len(df[df['rule3_result'] < 1]), len(df[df['rule3_result'] == 0]))
print(
    f"下浮率误差1%以内命中率：{round(len(df[df['rule3_result'] < 1]) / len(df), 2) * 100}%，下浮率命中率：{round(len(df[df['rule3_result'] == 0]) / len(df), 2) * 100}%")

df['rule4'] = round(df['rule1'] * 0.0 + df['rule2'] * 0.55 + df['rule3'] * 0.45, 2)
df['rule4_result'] = abs(df['下浮率'] - df['rule4'])
print(len(df[df['rule4_result'] < 1]), len(df[df['rule4_result'] == 0]))


def predict_KC(df, df_total):
    print('随机k值预测结果'.center(100, '*'))
    df['predict_k1'] = df['zbkzj'].apply(lambda x: huainan_kc(df_total, 'k1'))
    df['k1_result'] = abs(df['k1'] - df['predict_k1'])
    df['predict_k2'] = df['zbkzj'].apply(lambda x: huainan_kc(df_total, 'k2'))
    df['k2_result'] = abs(df['k2'] - df['predict_k2'])

    print(f"k1命中率：{round(len(df[df['k1_result'] == 0]) / len(df), 2) * 100}%")
    print(f"k2命中率：{round(len(df[df['k2_result'] == 0]) / len(df), 2) * 100}%")

    print(f"k1和k2命中率：{round(len(df[(df['k1_result'] == 0) & (df['k2_result'] == 0)]) / len(df), 2) * 100}%")


df[['k1', 'k2', 'zbkzj', 'kbjj']] = df[['k1', 'k2', 'zbkzj', 'kbjj']].fillna(0)

df = df[(df['k2'] < 1) &(df['k2'] != 0) & (df['k1'] != 0) & (df['zbkzj'] != 0) & (df['kbjj'] != 0)]


predict_KC(df,df)



# df = df[df['suck_factor'] != '']
def split_value(series):
    s = series['suck_factor']

    if not s:
        return '', '', '', '', ''
    s = re.sub(r'\s|;', '', s)
    group = re.findall("[抽取系组数情况球号为：:第]{6,}.{0,10}?([1234])[号组]", s)
    if not group:
        group = ['']
    m = re.findall("[Mm]值?.*?[值系数].*?(\d{1,2}\.?\d)", s)
    # kbjj = re.findall("[评标基准值：]{4,}(\d*.*\d*)元?",s)
    x1 = re.findall("[xX1]{2,}[，为对应系数值抽取x1值球号：号球，对应系数值:\d]+?(\d{1,2}\.\d)", s)
    x2 = re.findall("[xX2]{2,}[，为对应系数值抽取x1值球号：号球，对应系数值:\d]*?(\d{1,2}\.\d)", s)
    x3 = re.findall("[xX3]{2,}[，为对应系数值抽取x1值球号：号球，对应系数值:\d]*?(\d{1,2}\.\d)", s)
    if len(x1) == 3:
        x = x1
    else:
        x = x1[:1] + x2[:1] + x3[:1]
    # print(m,kbjj,x)
    # print(series['source_website_address'])
    print(group)
    print(s)
    if len(x) == 3:
        return group[0], m[0], x[0], x[1], x[2]
    else:
        x = x + (3 - len(x)) * ['']
        return group[0], m[0], x[0], x[1], x[2]


def extract_value():
    sql = "SELECT id,source_website_address,city,zbkzj,kbjj,k1,k2,project_type,publish_time,suck_factor FROM tender_bid_opening where city='池州市'"
    df = mysql_select_df(sql)
    df = df.fillna('')
    df['group'], df['m'], df['x1'], df['x2'], df['x3'] = zip(*df.apply(lambda x: split_value(x), axis=1))
    df['suck_factor'] = df['suck_factor'].apply(lambda s: re.sub(r'\s|;', '', s))
    df = df.fillna('')
    df = df[(df['m'] != '')]


def data_presentation():
    data = pd.read_csv('池州市抽取值数据.csv', encoding='gbk')
    data['publish_time'] = pd.to_datetime(data['publish_time'])
    data = data.sort_values(by=['publish_time'])
    # plt.plot(data['publish_time'], data[['m', 'x1', 'x2', 'x3', 'B']], label=['m', 'x1', 'x2', 'x3', 'B'])
    plt.plot(data['publish_time'], data['B'])
    # plt.legend()
    plt.show()


# data_presentation()

