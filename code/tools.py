import random
import re

import pandas as pd
from mysql_utils import *

from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor as XGBR
def train_model_predict(df,degree):
    # 提取数据
    df = df.dropna().reset_index(drop=True)
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=9)
    poly = PolynomialFeatures(degree=degree)

    x_train = poly.fit_transform(X_train.values)
    x_test = poly.fit_transform(X_test)

    # 线性回归
    la = Lasso(alpha=0.1, max_iter=100000)
    la.fit(x_train, y_train)
    print(f'线性回归训练集得分：{round(la.score(x_train, y_train), 2)}')
    print(f'线性回归测试集得分：{round(la.score(x_test, y_test), 2)}')

    # 随机森林回归
    rf = RandomForestRegressor(n_jobs=-1)
    rf.fit(x_train, y_train)
    print(f'随机森林回归训练集得分：{round(rf.score(x_train, y_train), 2)}')
    print(f'随机森林回归测试集得分：{round(rf.score(x_test, y_test), 2)}')

    # 决策树回归
    dt = DecisionTreeRegressor(max_depth=6)
    dt.fit(x_train, y_train)
    print(f'决策树回归训练集得分：{round(dt.score(x_train, y_train), 2)}')
    print(f'决策树回归测试集得分：{round(dt.score(x_test, y_test), 2)}')

    # K近邻回归
    kn = KNeighborsRegressor(n_neighbors=3, n_jobs=-1)
    kn.fit(x_train, y_train)
    print(f'k近邻回归测试集得分：{round(kn.score(x_test, y_test), 2)}')

    # XGBbost
    xgb = XGBR(n_estimators=100)
    xgb.fit(x_train, y_train)
    print(f"xgbbost测试集得分：{round(xgb.score(x_test,y_test),2)}") #你能想出这里应该返回什么模型评估指标么?

    return la, rf, dt, kn, xgb


def record_predict(df_train, x, degree):
    la, rf, dt, kn, xgb = train_model_predict(df_train,degree)

    poly = PolynomialFeatures(degree=degree)
    apply = np.array(x).reshape(1, -1)
    poly_apply = poly.fit_transform(apply)

    predict_list = [la.predict(poly_apply).tolist()[0], rf.predict(poly_apply).tolist()[0],
                    dt.predict(poly_apply).tolist()[0], kn.predict(poly_apply).tolist()[0],xgb.predict(poly_apply).tolist()[0]]

    return predict_list

def get_date(str_time: str):
    # str_time = 'http://lssggzy.lishui.gov.cn/art/2020/10/16/art_1229661956_138130.html'
    time = re.findall('(\d{4})/(\d{1,2})/(\d{1,2})', str_time)[0]
    s = "-".join(time)
    return s


def read_file(file_path):
    for ed in ['gbk', 'utf-8']:
        try:
            df = pd.read_csv(file_path, encoding=ed)
            break
        except:
            pass
    return df


def zt(d2):
    t3 = d2.groupby(['k1', 'k2'])['zbkzj'].count() \
        .reset_index(name='count') \
        .sort_values(['count'], ascending=False) \
        .head(5)
    # print(t3)
    return t3.max()


def get_prediction_interval(df, target_value):
    df['zbkzj'] = df['zbkzj'].astype(float)
    df = df.sort_values(by=['zbkzj'])
    max_value = df[df['zbkzj'] > target_value]
    max_value = max_value.iloc[:200] if len(max_value) >= 200 else max_value
    min_value = df[df['zbkzj'] < target_value]
    min_value = min_value.iloc[-200:] if len(min_value) >= 200 else min_value
    return pd.concat([max_value, min_value])


def rule1(df):
    length = int(len(df) * 0.3)
    df = df.sample(n=length)
    k1_list = df['下浮率'].values.tolist()
    if k1_list.count(max(k1_list, key=k1_list.count)) == 1:
        return round(max(k1_list, key=k1_list.count),2)
    else:
        return round(max(k1_list),2)


def rule2(df, target_value):
    # df = get_prediction_interval(df,target_value)
    value_list = df['下浮率'].values.tolist()
    length = int(len(df) * 0.4)
    df = df.sample(n=length)
    k1_list = df['下浮率'].values.tolist()
    k1_dict = {}
    for key in set(k1_list):
        k1_dict[key] = k1_list.count(key)
    return_k1 = 0
    for k, v in k1_dict.items():
        return_k1 += round(int(k) * int(v) / len(k1_list), 2)
    if return_k1 in value_list:
        return return_k1
    else:
        value_list = value_list+[return_k1]
        value_list = sorted(value_list)
        id = value_list.index(return_k1)
        if abs(return_k1-value_list[id+1]) < abs(return_k1-value_list[id-1]):
            return value_list[id+1]
        else:
            return value_list[id-1]

    # return return_k1


def rule3(df, target_value):
    df = get_prediction_interval(df, target_value)
    return round(df['下浮率'].mean() + random.uniform(-0.6, 0.6), 2)


def huainan_kc(df, choice):
    length = int(len(df)*0.2)
    df = df.sample(n=length)
    k1_list = df[choice].values.tolist()
    if k1_list.count(max(k1_list, key=k1_list.count)) == 1:
        predict = max(k1_list, key=k1_list.count)
    else:
        predict = max(k1_list)
    if predict in df[choice].values.tolist():
        return predict
    else:
        value_list = df[choice].values.tolist().append(predict)
        id = value_list.index(predict)
        if abs(predict-value_list[id+1]) < abs(predict-value_list[id-1]):
            return value_list[id+1]
        else:
            return value_list[id-1]


def luan_kc(df):
    length = int(len(df) * 0.1)
    df = df.sample(n=length)
    k1_list = df['k1'].values.tolist()
    if k1_list.count(max(k1_list, key=k1_list.count)) == 1:
        return max(k1_list, key=k1_list.count)
    else:
        return max(k1_list)


def lishui_data(file_path):
    # file_path = 'lishui.csv'
    df = read_file(file_path)
    df['zbkzj'] = df['zbkzj'].astype(float)
    df['kbjj'] = df['kbjj'].astype(float)

    df = df[df['zbkzj'] != 0]
    # df['Date'] = pd.to_datetime(df['Date'])

    # df['整数位'] = df['k1'].apply(lambda s: str(1000 - int(1000 * s))[:-1])
    # df['小数位'] = df['k1'].apply(lambda s: str(1000 - int(1000 * s))[-1])
    # df['下浮率'] = df['k1'].apply(lambda s: str(round(100 - 100 * s, 2)))
    # df['下浮率'] = df['下浮率'].astype(float)
    # df['下浮率'] = round(100 - 100 * df['kbjj'] / df['zbkzj'], 2)
    df = df[df['下浮率'] > 6]
    df = df[df['下浮率'] < 12]
    return df


def luan_data():
    df = read_file('luan.csv')
    df = df[df['zbkzj'] > df["kbjj"]]
    df['下浮率'] = 100 - round(df['kbjj'] / df['zbkzj'] * 100, 2)
    df = df[df['下浮率'] > 6]
    df = df[df['下浮率'] < 12]
    return df


def stage(s):
    for i in ["EPC", "总承包", "设计", "施工", "监理", "勘察", "造价", "咨询"]:
        if i in s:
            return i.replace("总承包", "EPC")
    return ""


def industy(s):
    for i in ['房屋建筑', '房建', '轻纺', '建设工程', '环保工程', '供电工程', '军工', '冶金', '商物粮', '核工业',
              '煤炭', '铁道', '化工石化医药', '电子通信', '机械', '建筑', '民航', '市政', '农林', '石油天然气', '公路',
              '电力', '水运', '建材', '水利', '海洋', '工程']:
        if i in s:
            return i.replace("房建", '房屋建筑')
    return ""


def huainan_data():
    df = read_file("huainan.csv")
    # df = df[['county','classify','k1','k2','zbkzj','kbjj','numbers_bidders','trade_method','project_type']]
    df[['k1', 'k2', 'zbkzj', 'kbjj', 'numbers_bidders']] = df[['k1', 'k2', 'zbkzj', 'kbjj', 'numbers_bidders']].fillna(0)
    df = df.fillna('')
    # df = pd.read_csv('huainan.csv', encoding='utf-8')

    # hy = [ '电子通信', '机械', '建筑', '民航', '市政', '农林', '公路', '电力', '水运', '建材', '水利', '海洋']

    df['stage'] = df['project_type'].apply(stage)
    df['industy'] = df['project_type'].apply(industy)

    # 审查字段分离情况
    # df = df[df['stage'] ==""]
    # df = df[df['industy'] ==""]

    # df['classify'] = enc.fit_transform(df['classify'].values.tolist())
    # print('classify',enc.inverse_transform(range(max(df['classify']))))

    # df['county'] = enc.fit_transform(df['county'].values.tolist())
    # print('county',enc.inverse_transform(range(max(df['county']))))

    # print(enc.inverse_transform([0,1,2,3]))
    df = df[(df['k1'] > 0) & (df['k2'] > 0) & (df['k1'] < 1) & (df['k2'] < 1)]
    # df = df.drop_duplicates()
    # df = df[df['k2']!=0]

    # df = df[df['k1']<1]
    # df = df[df['k2']<1]

    df = df[df['zbkzj'] > df["kbjj"]]
    df['下浮率'] = 100 - round(df['kbjj'] / df['zbkzj'] * 100, 2)
    df = df[df['下浮率'] < 20]

    for key in ['project_type']:  # 'county','classify','trade_method','stage','industy'
        enc = preprocessing.LabelEncoder()
        df[key] = enc.fit_transform(df[key].values.tolist())
        # print(key,enc.inverse_transform(range(max(df[key]))))
        for i, value in zip(range(max(df[key])), enc.inverse_transform(range(max(df[key])))):
            print(f"{key}：数字  {i}  代表  {value}")

    # df = df[['zbkzj', 'project_type', '下浮率', 'k1', 'k2']]  # 'county','classify','trade_method',,'stage'

    # df1 = df[df['project_type'] == '房屋建筑工程']
    # plt.plot(range(len(df1)),df1[['下浮率']],marker = 'o',label = ['下浮率','rule1','rule2'])
    # plt.legend()
    # plt.show()

    df.to_csv('淮南预测值对比.csv', index=False, encoding='utf-8')

    return df

def chizhou_data():
    sql = "SELECT id,source_website_address,city,zbkzj,kbjj,k1,k2,project_type FROM tender_bid_opening where city='池州市'"
    df = mysql_select_df(sql)
    df[['k1', 'k2', 'zbkzj', 'kbjj']] = df[['k1', 'k2', 'zbkzj', 'kbjj']].fillna(0)
    df[['zbkzj']] = df[['zbkzj']].astype(float)
    df[['kbjj']] = df[['kbjj']].astype(float)

    df = df.fillna('')
    df = df[(df['zbkzj'] > 100000) & (df['kbjj'] > 100000)]
    df = df[df['zbkzj'] > df["kbjj"]]
    df['下浮率'] = 100 - round(df['kbjj'] / df['zbkzj'] * 100, 2)
    df = df[df['下浮率'] < 20]
    project_label = {'水利水电工程': 1,  '公路工程': 2,  '市政公用工程': 3,  '建筑工程': 4,  '建筑装修装饰工程': 5}
    print(project_label)
    df['project_label'] = df['project_type'].apply(lambda s: project_label[s] if s in project_label.keys() else 6)
    # print(df)
    return  df
