# !/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import random
import time
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression as LR         # 逻辑回归
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor as XGBR
# 读取csv文件

# df = pd.read_csv('20230601.csv', encoding='utf-8')
df = pd.read_csv('lishui.csv', encoding='gbk')


d2 = df[['ZBKZJ','k1']]
d2 =d2[d2['k1'] !=0 ]

#提取数据
d1 = d2.dropna().reset_index(drop=True)
X = d1.drop(columns=['k1'])
y = d1['k1']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,shuffle=True,random_state=8)
poly = PolynomialFeatures(degree=1)


x_train = poly.fit_transform(X_train.values)
x_test = poly.fit_transform(X_test)


# 线性回归
la = Lasso(alpha=0.1,max_iter=100000)
la.fit(x_train,y_train)
print(f'线性回归训练集得分：{round(la.score(x_train,y_train),2)}')
print(f'线性回归测试集得分：{round(la.score(x_test,y_test),2)}')

# 随机森林回归
rf = RandomForestRegressor(n_jobs=-1)
rf.fit(x_train,y_train)
print(f'随机森林回归训练集得分：{round(rf.score(x_train,y_train),2)}')
print(f'随机森林回归测试集得分：{round(rf.score(x_test,y_test),2)}')

# 决策树回归
dt = DecisionTreeRegressor(max_depth = 6)
dt.fit(x_train,y_train)
print(f'决策树回归训练集得分：{round(dt.score(x_train,y_train),2)}')
print(f'决策树回归测试集得分：{round(dt.score(x_test,y_test),2)}')

# K近邻回归
kn = KNeighborsRegressor(n_neighbors=3,n_jobs=-1)
kn.fit(x_train,y_train)
print(f'k近邻回归测试集得分：{round(kn.score(x_test,y_test),2)}')

# XGBbost
reg = XGBR(n_estimators=100)
reg.fit(x_train,y_train)
print(f"xgbbost测试集得分：{round(reg.score(x_test,y_test),2)}") #你能想出这里应该返回什么模型评估指标么?

apply = np.array([19396110.3]).reshape(1,-1)
#apply = np.array([67583073.37,0.94,0.4]).reshape(1,-1)
poly_apply = poly.fit_transform(apply)
r1=la.predict(poly_apply)
r2=rf.predict(poly_apply)
print(la.predict(poly_apply))
print(dt.predict(poly_apply))
print(kn.predict(poly_apply))
print(rf.predict(poly_apply))

print(reg.predict(poly_apply)) #传统接口predict

arrayR=[la.predict(poly_apply),rf.predict(poly_apply),dt.predict(poly_apply),kn.predict(poly_apply)]
df1=pd.DataFrame(arrayR)
maximum = df1.max()[0]
minimum = df1.min()[0]
total=(la.predict(poly_apply))+(rf.predict(poly_apply))+(dt.predict(poly_apply))+(kn.predict(poly_apply))
res=(total-maximum-minimum)/2
print(res)

print(sum([rf.predict(poly_apply),kn.predict(poly_apply),reg.predict(poly_apply)])/3)

# end = time.time()
# from sklearn.metrics import accuracy_score
# print(accuracy_score(y_test,r1))
# print(accuracy_score(y_test,r2))

#print(X)

#拆分数据集


#预测算法
