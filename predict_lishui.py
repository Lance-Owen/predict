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
        
    d2 = df[['zbkzj','k1','k2','kbjj']]

    #提取数据
    d1 = d2.dropna().reset_index(drop=True)
    X = d1.drop(columns=['kbjj'])
    y = d1['kbjj']
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
    # print(f"xgbbost测试集得分：{round(xgb.score(x_test,y_test),2)}") #你能想出这里应该返回什么模型评估指标么?

    return la,rf,dt,kn,xgb

def record_predict(df_train,KZJ):
    predict_k1 = zt(df.sample(n=8))['k1']
    predict_k2 = zt(df.sample(n=8))['k2']


    la,rf,dt,kn,xgb = train_model_predict(df_train)

    poly = PolynomialFeatures(degree=1)
    apply = np.array([KZJ,predict_k1,predict_k2]).reshape(1,-1)
    poly_apply = poly.fit_transform(apply)

    predict_list = [la.predict(poly_apply).tolist()[0],rf.predict(poly_apply).tolist()[0],dt.predict(poly_apply).tolist()[0],kn.predict(poly_apply).tolist()[0]]

    res=(sum(predict_list)-max(predict_list)-min(predict_list))/2

    xgb_mean = sum([rf.predict(poly_apply),kn.predict(poly_apply),xgb.predict(poly_apply)])/3
    return la.predict(poly_apply).tolist()[0],rf.predict(poly_apply).tolist()[0],dt.predict(poly_apply).tolist()[0],kn.predict(poly_apply).tolist()[0],res#,xgb.predict(poly_apply).tolist()[0],xgb_mean[0]



### 丽水市
df_total = lishui_data('lishui.csv')
file = '丽水市测试数据0629.csv'
df = lishui_data(file)

df = df_total
print(len(df))

df['rule1'] = df['zbkzj'].apply(lambda x: rule1(df_total))
df['rule1_result'] = abs(df['下浮率'] - df['rule1'])
print(len(df[df['rule1_result'] < 1]),len(df[df['rule1_result'] == 0]))

df['rule2'] = df['zbkzj'].apply(lambda x: rule2(df_total,x))
df['rule2_result'] = abs(df['下浮率'] - df['rule2'])
print(len(df[df['rule2_result'] < 1]),len(df[df['rule2_result'] == 0]))

print(f"下浮率误差1%以内命中率：{round(len(df[df['rule2_result'] < 1])/len(df),2)*100}%，下浮率命中率：{round(len(df[df['rule2_result'] == 0])/len(df),2)*100}%")

df['rule3'] = df['zbkzj'].apply(lambda x: rule3(df_total, x))
df['rule3_result'] = abs(df['下浮率'] - df['rule3'])
print(len(df[df['rule3_result'] < 1]),len(df[df['rule3_result'] == 0]))
print(f"下浮率误差1%以内命中率：{round(len(df[df['rule3_result'] < 1])/len(df),2)*100}%，下浮率命中率：{round(len(df[df['rule3_result'] == 0])/len(df),2)*100}%")

df['rule4'] = round(df['rule1']*0.3 + df['rule2']*0.4 + df['rule3']*0.3,2)
df['rule4_result'] = abs(df['下浮率'] - df['rule4'])
print(len(df[df['rule4_result'] < 1]), len(df[df['rule4_result'] == 0]))


# df.to_csv(f'{file.split(".")[0]}_结果.csv', index=False)

# plt.plot(range(len(df)),df[['下浮率','rule3','rule2']],marker = 'o',label = ['下浮率','rule1','rule2'])
# plt.legend()
# plt.show()
# df.to_csv('lishui测试2.csv', index=False)


# plt.hist(df['下浮率'], bins=20)









#
#
# #导入网格搜索模块
# from sklearn.model_selection import GridSearchCV
# rfr_best = RandomForestRegressor()
# params ={'n_estimators':range(10,20,1)}
# gs = GridSearchCV(rfr_best, params, cv=4)
# gs.fit(X_train,Y_train)
#
# #查验优化后的超参数配置
# print(gs.best_score_)
# print(gs.best_params_)
#
#
#














# df_test[['线性回归', "随机森林",'决策树','K近邻','XGB','预测值']] = df_test.apply(lambda x:record_predict(df_train,x['ZBKZJ']),axis=1,result_type='expand')
# df_train[['线性回归', "随机森林",'决策树','K近邻','XGB','预测值']] = df_train.apply(lambda x:record_predict(df_train,x['ZBKZJ']),axis=1,result_type='expand')




# df_test.to_csv('预测值对比.csv',index=False)

# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.plot(range(len(df_test)),df_test[['k1','线性回归', "随机森林",'决策树','K近邻','XGB','预测值']],marker = 'o',label = ['k1','线性回归', "随机森林",'决策树','K近邻','XGB','预测值'])
# plt.legend()
# plt.show()

# chatgpt_k1 = [8.3, 7.6, 9.1, 8.8, 7.9, 8.5, 7.2, 9.0, 8.2, 8.6, 7.7, 8.4, 8.1, 7.8, 8.0, 7.5, 8.7, 7.4, 8.9, 7.3, 8.8, 7.6, 9.2, 8.3, 7.9, 9.1, 8.5, 7.7, 9.0, 8.6, 7.8, 8.7, 7.5, 8.9, 7.4, 8.8, 7.6, 9.3, 8.4, 7.9, 9.2, 8.5, 7.7, 9.1, 8.6, 7.8, 9.0, 8.7, 7.5, 9.2]
# df_test['chatgpt_k1'] = chatgpt_k1[:30]

# plt.plot(range(len(df_test)),df_test[['下浮率', "chatgpt_k1"]],marker = 'o')

# plt.sc

