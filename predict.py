import pandas as pd
import numpy as np
import pandas as pd
from tools import *
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


def train_model_predict(df):
        
    d2 = df[['ZBKZJ','k1']]
    d2 =d2[d2['k1'] !=0 ]

    #提取数据
    d1 = d2.dropna().reset_index(drop=True)
    X = d1.drop(columns=['k1'])
    y = d1['k1']
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,shuffle=True,random_state=8)
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
    xgb = XGBR(n_estimators=100)
    xgb.fit(x_train,y_train)
    print(f"xgbbost测试集得分：{round(xgb.score(x_test,y_test),2)}") #你能想出这里应该返回什么模型评估指标么?

    return la,rf,dt,kn,xgb

    apply = np.array([KZJ,k1]).reshape(1,-1)
    #apply = np.array([67583073.37,0.94,0.4]).reshape(1,-1)
    poly_apply = poly.fit_transform(apply)
    # r1=la.predict(poly_apply)
    # r2=rf.predict(poly_apply)
    # # print(la.predict(poly_apply))
    # # print(dt.predict(poly_apply))
    # # print(kn.predict(poly_apply))
    # # print(rf.predict(poly_apply))

    # # print(reg.predict(poly_apply)) #传统接口predict

    # arrayR=[la.predict(poly_apply),rf.predict(poly_apply),dt.predict(poly_apply),kn.predict(poly_apply)]
    # df1=pd.DataFrame(arrayR)
    # maximum = df1.max()[0]
    # minimum = df1.min()[0]
    # total=(la.predict(poly_apply))+(rf.predict(poly_apply))+(dt.predict(poly_apply))+(kn.predict(poly_apply))
    # res=(total-maximum-minimum)/2

    # return res
    # print(res)

    # print(sum([rf.predict(poly_apply),kn.predict(poly_apply),reg.predict(poly_apply)])/3)

def record_predict(df_train,KZJ):
    # predict_k1 = zt(df.sample(n=8))['k1']
    # predict_k1 = df_test.iloc[0]['k1']

    # KZJ = df_test.iloc[0]['ZBKZJ']
    # kbjj = df_test.iloc[0]['kbjj']


    la,rf,dt,kn,xgb = train_model_predict(df_train)

    poly = PolynomialFeatures(degree=1)
    apply = np.array([KZJ]).reshape(1,-1)
    poly_apply = poly.fit_transform(apply)

    predict_list = [la.predict(poly_apply).tolist()[0],rf.predict(poly_apply).tolist()[0],dt.predict(poly_apply).tolist()[0],kn.predict(poly_apply).tolist()[0]]

    res=(sum(predict_list)-max(predict_list)-min(predict_list))/2

    # return predict_list+[predict_k1,res]
    return la.predict(poly_apply).tolist()[0],rf.predict(poly_apply).tolist()[0],dt.predict(poly_apply).tolist()[0],kn.predict(poly_apply).tolist()[0],xgb.predict(poly_apply).tolist()[0],res




df = lishui_data()
zt(df.sample(n=500))['k1']
df['Date'] = pd.to_datetime(df['Date'])
df['整数位'] = df['k1'].apply(lambda s:str(1000-int(1000*s))[:-1])
df['小数位'] = df['k1'].apply(lambda s:str(1000-int(1000*s))[-1])

df_train = df[df['Date']<'2023-06'][['ZBKZJ','k1']]
df_test = df[df['Date']>='2023-06'][['ZBKZJ','k1']]



df_test[['线性回归', "随机森林",'决策树','K近邻','XGB','预测值']] = df_test.apply(lambda x:record_predict(df_train,x['ZBKZJ']),axis=1,result_type='expand')
df_train[['线性回归', "随机森林",'决策树','K近邻','XGB','预测值']] = df_train.apply(lambda x:record_predict(df_train,x['ZBKZJ']),axis=1,result_type='expand')



def rand_k1(s):
    print(i)
    return zt(df_train.sample(n=i))['k1']
x = []
for i in range(100,300):
    df_test['随机数'] = df_test.apply(rand_k1,axis=1)
    df_test['差值'] = (df_test['k1']-df_test['随机数'])**2
    x.append(df_test['差值'].values.sum()/30)


df_test.to_csv('预测值对比.csv',index=False)

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.plot(range(len(df_test)),df_test[['k1','线性回归', "随机森林",'决策树','K近邻','XGB','预测值']],marker = 'o',label = ['k1','线性回归', "随机森林",'决策树','K近邻','XGB','预测值'])
plt.legend()
plt.show()

chatgpt_k1 = [8.3, 7.6, 9.1, 8.8, 7.9, 8.5, 7.2, 9.0, 8.2, 8.6, 7.7, 8.4, 8.1, 7.8, 8.0, 7.5, 8.7, 7.4, 8.9, 7.3, 8.8, 7.6, 9.2, 8.3, 7.9, 9.1, 8.5, 7.7, 9.0, 8.6, 7.8, 8.7, 7.5, 8.9, 7.4, 8.8, 7.6, 9.3, 8.4, 7.9, 9.2, 8.5, 7.7, 9.1, 8.6, 7.8, 9.0, 8.7, 7.5, 9.2]
df_test['chatgpt_k1'] = chatgpt_k1[:30]
df_test['下浮率'] = df_test['k1'].apply(lambda s:str(round(100-100*s,2)))
plt.plot(range(len(df_test)),df_test[['下浮率', "chatgpt_k1"]],marker = 'o')




# x.index(max(x))
x.index(min(x))




# 遍历所有数据，得到KZJ
#     生成k1
#     训练模型做预测，但是每调用一次，模型会变化一次，当生成两个k的时候，调用两次会模型训练数据效果不一致，期望同一个模型，不同k


