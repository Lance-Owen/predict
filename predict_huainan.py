from tools import *
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor as XGBR
import numpy as np
from tools import *


def train_model_predict(df):
        
    # d2 = df[['zbkzj','k1','k2','kbjj']]

    #提取数据
    d1 = df.dropna().reset_index(drop=True)
    X = d1.drop(columns=['下浮率'])
    y = d1['下浮率']
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,shuffle=True,random_state=9)
    poly = PolynomialFeatures(degree=1)


    x_train = poly.fit_transform(X_train.values)
    x_test = poly.fit_transform(X_test)


    # 线性回归
    la = Lasso(alpha=0.1,max_iter=100000)
    la.fit(x_train,y_train)
    # print(f'线性回归训练集得分：{round(la.score(x_train,y_train),2)}')
    # print(f'线性回归测试集得分：{round(la.score(x_test,y_test),2)}')

    # 随机森林回归
    rf = RandomForestRegressor(n_jobs=-1)
    rf.fit(x_train,y_train)
    # print(f'随机森林回归训练集得分：{round(rf.score(x_train,y_train),2)}')
    # print(f'随机森林回归测试集得分：{round(rf.score(x_test,y_test),2)}')

    # 决策树回归
    dt = DecisionTreeRegressor(max_depth = 6)
    dt.fit(x_train,y_train)
    # print(f'决策树回归训练集得分：{round(dt.score(x_train,y_train),2)}')
    # print(f'决策树回归测试集得分：{round(dt.score(x_test,y_test),2)}')

    # K近邻回归
    kn = KNeighborsRegressor(n_neighbors=3,n_jobs=-1)
    kn.fit(x_train,y_train)
    # print(f'k近邻回归测试集得分：{round(kn.score(x_test,y_test),2)}')

    # XGBbost
    xgb = XGBR(n_estimators=100)
    xgb.fit(x_train,y_train)
    # print(f"xgbbost测试集得分：{round(xgb.score(x_test,y_test),2)}") #你能想出这里应该返回什么模型评估指标么?

    return la,rf,dt,kn,xgb


def record_predict(df_train,predict_value):

    # predict_k1 = zt(df.sample(n=8))['k1']
    # predict_k2 = zt(df.sample(n=8))['k2']


    la,rf,dt,kn,xgb = train_model_predict(df_train)

    poly = PolynomialFeatures(degree=1)
    apply = np.array(predict_value).reshape(1,-1)
    poly_apply = poly.fit_transform(apply)

    predict_list = [la.predict(poly_apply).tolist()[0],rf.predict(poly_apply).tolist()[0],dt.predict(poly_apply).tolist()[0],kn.predict(poly_apply).tolist()[0]]

    res=(sum(predict_list)-max(predict_list)-min(predict_list))/2

    xgb_mean = sum([rf.predict(poly_apply),kn.predict(poly_apply),xgb.predict(poly_apply)])/3
    return la.predict(poly_apply).tolist()[0],rf.predict(poly_apply).tolist()[0],dt.predict(poly_apply).tolist()[0],kn.predict(poly_apply).tolist()[0],res#,xgb.predict(poly_apply).tolist()[0],xgb_mean[0]





### 淮南市 

# df[['zbkzj','county','classify','trade_method','stage','industy','下浮率']]

df = huainan_data()

county = 0
classify = 0
trade_method = 0
stage = 2
industy = 7

predict_value = [7140000,2]
print(record_predict(df, predict_value))

sum_k = 0
for i in range(20):
    sum_k += record_predict(df, predict_value)[-1]

print(sum_k/20)

import pandas as pd
data = pd.read_csv('淮南市一周数据.csv')

data['预测下浮率'] = data.apply(lambda x: record_predict(df,[x[['zbkzj']],x['project_type']]))