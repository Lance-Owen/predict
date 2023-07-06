from tools import *

df_total = chizhou_data()

df = df_total

print(len(df))

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
    df[['k1', 'k2', 'zbkzj', 'kbjj']] = df[['k1', 'k2', 'zbkzj', 'kbjj']].astype(float)
    df[['k1', 'k2', 'zbkzj', 'kbjj']] = df[['k1', 'k2', 'zbkzj', 'kbjj']].fillna(0)
    df = df[(df['k1'] > 0) & (df['k2'] > 0) & (df['k1'] < 1) & (df['k2'] < 1)]
    df = df[df['zbkzj'] > df["kbjj"]]
    df['下浮率'] = 100 - round(df['kbjj'] / df['zbkzj'] * 100, 2)
    df = df[df['下浮率'] < 20]

    df[['k1', 'k2', 'zbkzj', 'kbjj', 'numbers_bidders']] = df[['k1', 'k2', 'zbkzj', 'kbjj', 'numbers_bidders']].fillna(
        0)
    df = df.fillna('')

    project_label = {'水利水电工程': 1,  '公路工程': 2,  '市政公用工程': 3,  '建筑工程': 4,  '建筑装修装饰工程': 5}
    df['project_label'] = df['project_type'].apply(lambda s: project_label[s] if s in project_label.keys() else 6)

    df['预测下浮率'] = df.apply(lambda x: record_predict(df_train, [x['zbkzj'], x['project_label']])[-1], axis=1)

    df['result'] = abs(df['下浮率'] - df['预测下浮率'])
    print('机器学习模型预测结果'.center(100, '*'))
    print(
        f"下浮率误差1%以内命中率：{round(len(df[df['result'] < 1]) / len(df), 2) * 100}%，下浮率命中率：{round(len(df[df['result'] == 0]) / len(df), 2) * 100}%")


print(set(df['project_type'].values.tolist()))

df_train = df[['zbkzj','下浮率','project_label']]
# df['预测下浮率'] = df.apply(lambda x: record_predict(df_train, [x['zbkzj'],3])[-1], axis=1)
# df['result'] = abs(df['下浮率'] - df['预测下浮率'])
# print('机器学习模型预测结果'.center(100, '*'))
# print(
#     f"下浮率误差1%以内命中率：{round(len(df[df['result'] < 1]) / len(df), 2) * 100}%，下浮率命中率：{round(len(df[df['result'] == 0]) / len(df), 2) * 100}%")

print(record_predict(df_train, [9216194.55,1])[-1])
# df.to_csv('池州市数据.csv',index=False)


#
# df['rule1'] = df['zbkzj'].apply(lambda x: rule1(df_total))
# df['rule1_result'] = abs(df['下浮率'] - df['rule1'])
# print(len(df[df['rule1_result'] < 1]),len(df[df['rule1_result'] == 0]))
#
# df['rule2'] = df['zbkzj'].apply(lambda x: rule2(df_total, x))
# df['rule2_result'] = abs(df['下浮率'] - df['rule2'])
# print(len(df[df['rule2_result'] < 1]), len(df[df['rule2_result'] == 0]))
# print(f"下浮率误差1%以内命中率：{round(len(df[df['rule2_result'] < 1])/len(df),2)*100}%，下浮率命中率：{round(len(df[df['rule2_result'] == 0])/len(df),2)*100}%")
#
# df['rule3'] = df['zbkzj'].apply(lambda x: rule3(df_total, x))
# df['rule3_result'] = abs(df['下浮率'] - df['rule3'])
# print(len(df[df['rule3_result'] < 1]), len(df[df['rule3_result'] == 0]))
# print(f"下浮率误差1%以内命中率：{round(len(df[df['rule3_result'] < 1])/len(df),2)*100}%，下浮率命中率：{round(len(df[df['rule3_result'] == 0])/len(df),2)*100}%")
#
# df['rule4'] = round(df['rule1']*0.3 + df['rule2']*0.4 + df['rule3']*0.3,2)
# df['rule4_result'] = abs(df['下浮率'] - df['rule4'])
# print(len(df[df['rule4_result'] < 1]), len(df[df['rule4_result'] == 0]))


# df.to_csv('丽水预测结果.csv', index=False)
